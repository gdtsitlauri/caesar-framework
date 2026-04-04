"""
ADPN — Adaptive Defense Policy Network
=======================================
Novel Contribution #2 of the CAESAR framework.

Architecture: Dueling Double-DQN
  • Dueling streams  → separate value V(s) and advantage A(s,a) heads
  • Double DQN       → decouple action selection from action evaluation
  • Prioritised-like experience replay buffer

The defender receives reward signals from the environment AND
co-evolutionary feedback from the threat graph (proactive signal),
allowing it to pre-empt attacks before they fully materialise.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Tuple


# ─── Hyper-parameters ────────────────────────────────────────────────
LR          = 1e-3
GAMMA       = 0.95
EPSILON_INI = 1.00
EPSILON_MIN = 0.05
EPSILON_DEC = 0.995
BATCH_SIZE  = 64
MEM_SIZE    = 12_000
UPDATE_TGT  = 100


# ═══════════════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    """
    Dueling DQN:  Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

    Separate value / advantage streams share a common feature extractor.
    This makes the network faster at learning which *states* are
    (un)favourable, independent of which action is chosen.
    """

    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f  = self.feature(x)
        v  = self.value(f)
        a  = self.advantage(f)
        return v + a - a.mean(dim=-1, keepdim=True)


# ═══════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """Circular replay buffer — stores (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int = MEM_SIZE):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_, done):
        self.buf.append((
            np.asarray(s,  dtype=np.float32),
            int(a),
            float(r),
            np.asarray(s_, dtype=np.float32),
            float(done),
        ))

    def sample(self, k: int):
        batch = random.sample(self.buf, k)
        s, a, r, s_, d = zip(*batch)
        return (
            np.array(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s_),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ═══════════════════════════════════════════════════════════════════════
class ADPN:
    """
    Adaptive Defense Policy Network — Dueling Double-DQN agent.

    The defender observes the current network state and selects one of
    N_DEFENSE_ACTIONS countermeasures to deploy.

    Proactive override
    ──────────────────
    When the Temporal Attack Graph predicts an upcoming attack with high
    confidence, the CAESAR loop passes a `proactive_action` override.
    The ADPN still trains on the overridden transitions so it learns
    from proactive deployments over time.
    """

    def __init__(self, state_dim: int, n_actions: int, device: str = 'cpu'):
        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self.device     = device

        self.policy_net = DuelingDQN(state_dim, n_actions).to(device)
        self.target_net = DuelingDQN(state_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory     = ReplayBuffer(MEM_SIZE)

        self.epsilon    = EPSILON_INI
        self._step      = 0

        # history
        self.losses:  list = []
        self.rewards: list = []

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray,
                      proactive: Optional[int] = None) -> int:
        """ε-greedy action selection with optional proactive override."""
        if proactive is not None:
            return proactive
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(s).argmax(dim=1).item())

    # ------------------------------------------------------------------
    def store(self, s, a, r, s_, done):
        self.memory.push(s, a, r, s_, done)
        self.rewards.append(float(r))

    # ------------------------------------------------------------------
    def train_step(self) -> Optional[float]:
        """One gradient update step. Returns loss or None if buffer small."""
        if len(self.memory) < BATCH_SIZE:
            return None

        s, a, r, s_, d = self.memory.sample(BATCH_SIZE)
        S  = torch.FloatTensor(s).to(self.device)
        A  = torch.LongTensor(a).to(self.device)
        R  = torch.FloatTensor(r).to(self.device)
        S_ = torch.FloatTensor(s_).to(self.device)
        D  = torch.FloatTensor(d).to(self.device)

        # Current Q-values
        q_curr = self.policy_net(S).gather(1, A.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            a_next    = self.policy_net(S_).argmax(dim=1)
            q_next    = self.target_net(S_).gather(1, a_next.unsqueeze(1)).squeeze(1)
            q_target  = R + GAMMA * q_next * (1.0 - D)

        loss = F.smooth_l1_loss(q_curr, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DEC)
        self._step  += 1

        # Periodic target network sync
        if self._step % UPDATE_TGT == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        lv = loss.item()
        self.losses.append(lv)
        return lv

    # ------------------------------------------------------------------
    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions (used for analysis/vis)."""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(s).cpu().numpy().flatten()
