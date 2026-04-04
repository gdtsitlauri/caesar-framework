"""
CAESAR — Co-Evolutionary Adversarial Simulation Engine for Attack & Response
=============================================================================
Core Algorithm (Novel Contribution #4)

Formal algorithm
────────────────
Input : CyberEnvironment E, initial policies π_θ (ADPN), G_φ/D_ψ (TA-GAN),
        empty Temporal Attack Graph TAG

At each episode step t:
  1. d_t  ← E.get_defense_embedding()              # observe defense state
  2. (k,ι)← G_φ(z~𝒩, d_t)                         # TA-GAN generates attack
  3. r_a  ← E.inject_attack(k, ι)                  # measure attack success
  4. p_t  ← TAG.get_proactive_defense(history)      # graph-based preemption
  5. a_t  ← π_θ(s_t, proactive=p_t)                # defender selects action
  6. r_d  ← E.apply_defense(a_t)                    # measure defense reward
  7. TAG.add_event(k, a_t, r_a, r_d)               # update knowledge graph
  8. π_θ  ← ADPN.train_step(s_t, a_t, r_d, s_{t+1})
  9. G_φ/D_ψ← TA-GAN.train_step(real_buf, d_t)
 10. G_φ  ← update bypass_reward(r_a)             # co-evolutionary signal
 11. F_att← r_a × (1 - 0.3·r_d)                  # attacker fitness
     F_def← r_d × (1 - 0.3·r_a) + 0.2·health    # defender fitness

Novelty summary
───────────────
• Steps 2–3 : Defense-conditioned generation (TA-GAN)
• Step 4    : Proactive preemption via graph prediction
• Step 10   : Closed-loop co-evolutionary feedback
• Steps 11  : Bi-directional fitness metrics
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

from .environment  import CyberEnvironment
from .ta_gan       import ThreatAwareGAN
from .adpn         import ADPN
from .threat_graph import TemporalAttackGraph


# ═══════════════════════════════════════════════════════════════════════
class CAESAR:
    """
    CAESAR framework — ties together TA-GAN, ADPN, and Temporal Attack Graph
    in a co-evolutionary training loop.
    """

    VERSION = "1.0.0"

    def __init__(self,
                 env:    CyberEnvironment,
                 device: str = 'cpu'):
        self.env    = env
        self.device = device

        self.ta_gan = ThreatAwareGAN(device=device)
        self.adpn   = ADPN(
            state_dim=env.STATE_DIM,
            n_actions=env.N_DEFENSE_ACTIONS,
            device=device,
        )
        self.tag = TemporalAttackGraph()

        # Running history
        self.episode_logs:       List[Dict] = []
        self.attacker_fitness:   List[float] = []
        self.defender_fitness:   List[float] = []
        self._recent_attacks:    deque = deque(maxlen=10)
        self.total_proactive:    int   = 0
        self.total_steps:        int   = 0
        self._gan_buf:           List  = []   # mini-buffer for GAN training

        self._start_time: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────
    def run_episode(self,
                    n_steps: int = 50,
                    train:   bool = True) -> Dict:
        """
        Run one full episode.
        Returns episode metric dictionary.
        """
        state = self.env.reset()

        ep_defense_reward   = 0.0
        ep_attack_success   = 0.0
        ep_detection        = 0.0
        ep_proactive        = 0
        ep_fpr              = 0.0

        for step in range(n_steps):
            # ── 1. Observe defense state ────────────────────────────
            d_emb = self.env.get_defense_embedding()

            # ── 2-3. Attack generation & injection ──────────────────
            atk_type, intensity = self.ta_gan.generate_attack(d_emb)
            atk_success = self.env.inject_attack(atk_type, intensity)
            self._recent_attacks.append(atk_type)

            # ── 4. Proactive defense via TAG ────────────────────────
            proactive = self.tag.get_proactive_defense(list(self._recent_attacks))
            if proactive is not None:
                ep_proactive += 1

            # ── 5. Defender action selection ────────────────────────
            new_state = self.env.get_state()
            action = self.adpn.select_action(new_state, proactive=proactive)

            # ── 6. Apply defense & collect reward ───────────────────
            def_reward, fpr = self.env.apply_defense(action)
            next_state = self.env.get_state()
            done = (step == n_steps - 1)

            # ── 7. TAG update ────────────────────────────────────────
            self.tag.add_event(
                attack_type=atk_type,
                defense_action=action,
                attack_success=atk_success,
                defense_reward=def_reward,
                step=self.total_steps,
            )

            # ── 8-10. Learning (train mode only) ────────────────────
            if train:
                # ADPN learning
                self.adpn.store(new_state, action, def_reward, next_state, done)
                self.adpn.train_step()

                # TA-GAN learning
                self._gan_buf.append(new_state)
                if len(self._gan_buf) >= 32:
                    real_batch = np.array(self._gan_buf[:32])
                    self.ta_gan.train_step(real_batch, d_emb)
                    self._gan_buf = self._gan_buf[32:]

                # Co-evolutionary bypass feedback
                self.ta_gan.update_bypass_reward(atk_success)

            # ── Metric accumulation ──────────────────────────────────
            ep_defense_reward += def_reward
            ep_attack_success += atk_success
            ep_detection      += self.env.state.detection_conf
            ep_fpr            += fpr
            self.total_steps  += 1
            state              = next_state

        # ── 11. Episode fitness ──────────────────────────────────────
        T   = n_steps
        avg_dr  = ep_defense_reward / T
        avg_as  = ep_attack_success / T
        avg_det = ep_detection      / T
        avg_fpr = ep_fpr            / T
        health  = self.env.network_health()

        f_att = avg_as  * (1 - 0.30 * avg_dr)
        f_def = avg_dr  * (1 - 0.30 * avg_as) + 0.20 * health

        self.attacker_fitness.append(f_att)
        self.defender_fitness.append(f_def)
        self.total_proactive += ep_proactive

        log = {
            'avg_defense_reward':  avg_dr,
            'avg_attack_success':  avg_as,
            'avg_detection_rate':  avg_det,
            'avg_fpr':             avg_fpr,
            'network_health':      health,
            'attacker_fitness':    f_att,
            'defender_fitness':    f_def,
            'n_proactive':         ep_proactive,
            'epsilon':             self.adpn.epsilon,
            'graph_summary':       self.tag.summary(),
            'bypass_reward':       self.ta_gan.bypass_reward,
        }
        self.episode_logs.append(log)
        return log

    # ──────────────────────────────────────────────────────────────────
    def train(self,
              n_episodes: int  = 150,
              n_steps:    int  = 50,
              verbose:    bool = True) -> List[Dict]:
        """Full co-evolutionary training loop."""
        self._start_time = time.time()
        print("=" * 65)
        print(f"  CAESAR v{self.VERSION} — Co-evolutionary Training")
        print(f"  Episodes: {n_episodes}  |  Steps/ep: {n_steps}  |  Device: {self.device}")
        print("=" * 65)

        for ep in range(1, n_episodes + 1):
            log = self.run_episode(n_steps=n_steps, train=True)

            if verbose and ep % 10 == 0:
                elapsed = time.time() - self._start_time
                print(
                    f"  EP {ep:4d}/{n_episodes}"
                    f"  | DefReward {log['avg_defense_reward']:+.3f}"
                    f"  | AtkSuccess {log['avg_attack_success']:.3f}"
                    f"  | Health {log['network_health']:.3f}"
                    f"  | ε {log['epsilon']:.3f}"
                    f"  | Proact {log['n_proactive']:2d}"
                    f"  | {elapsed:.0f}s"
                )

        elapsed = time.time() - self._start_time
        print("=" * 65)
        print(f"  Training complete in {elapsed:.1f}s")
        print(f"  Total proactive defenses: {self.total_proactive}")
        print(f"  Graph: {self.tag.summary()}")
        print("=" * 65)
        return self.episode_logs

    # ──────────────────────────────────────────────────────────────────
    def evaluate(self,
                 n_episodes: int = 20,
                 n_steps:    int = 50) -> Dict:
        """Evaluate without training (greedy policy)."""
        print(f"\n  Evaluating over {n_episodes} episodes...")
        old_eps = self.adpn.epsilon
        self.adpn.epsilon = 0.0   # fully greedy

        logs = [self.run_episode(n_steps=n_steps, train=False)
                for _ in range(n_episodes)]

        self.adpn.epsilon = old_eps

        scalar_keys = [k for k, v in logs[0].items() if isinstance(v, (int, float))]
        result = {k: float(np.mean([m[k] for m in logs])) for k in scalar_keys}

        print("  Evaluation results:")
        for k, v in result.items():
            print(f"    {k:<28s}: {v:.4f}")
        return result

    # ──────────────────────────────────────────────────────────────────
    def save(self, path: str = 'checkpoints/'):
        """Save model weights and logs."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.ta_gan.G.state_dict(), f"{path}/ta_gan_G.pt")
        torch.save(self.ta_gan.D.state_dict(), f"{path}/ta_gan_D.pt")
        torch.save(self.adpn.policy_net.state_dict(), f"{path}/adpn_policy.pt")
        with open(f"{path}/episode_logs.json", 'w') as f:
            json.dump(self.episode_logs, f, indent=2, default=str)
        print(f"  Saved checkpoint to {path}/")

    def load(self, path: str = 'checkpoints/'):
        """Load model weights."""
        self.ta_gan.G.load_state_dict(torch.load(f"{path}/ta_gan_G.pt", map_location=self.device))
        self.ta_gan.D.load_state_dict(torch.load(f"{path}/ta_gan_D.pt", map_location=self.device))
        self.adpn.policy_net.load_state_dict(torch.load(f"{path}/adpn_policy.pt", map_location=self.device))
        print(f"  Loaded checkpoint from {path}/")
