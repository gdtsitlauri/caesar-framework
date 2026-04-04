"""
CAESAR Framework — Network Simulation Environment
==================================================
Simulates a realistic network topology with hosts, servers, routers, firewalls.
Supports attack injection, defense deployment, and state tracking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from enum import IntEnum


class AttackType(IntEnum):
    NORMAL       = 0
    DOS          = 1
    DDOS         = 2
    PORT_SCAN    = 3
    BRUTE_FORCE  = 4
    DATA_EXFIL   = 5
    MITM         = 6
    RANSOMWARE   = 7


class DefenseAction(IntEnum):
    NO_ACTION           = 0
    BLOCK_IP            = 1
    RATE_LIMIT          = 2
    ISOLATE_NODE        = 3
    DEPLOY_HONEYPOT     = 4
    ALERT_ADMIN         = 5
    PATCH_VULNERABILITY = 6
    RESET_CONNECTION    = 7


ATTACK_NAMES  = [a.name for a in AttackType]
DEFENSE_NAMES = [d.name for d in DefenseAction]


@dataclass
class NetworkNode:
    node_id:           int
    node_type:         str   # 'host', 'server', 'router', 'firewall'
    is_compromised:    bool  = False
    vulnerability:     float = 0.5
    traffic_load:      float = 0.0
    patched:           bool  = False


@dataclass
class NetworkState:
    n_nodes:             int   = 10
    packet_rate:         float = 0.0
    anomaly_score:       float = 0.0
    attack_type:         int   = 0
    attack_intensity:    float = 0.0
    defense_active:      List[int] = field(default_factory=list)
    compromised_nodes:   int   = 0
    detection_conf:      float = 0.0
    false_positive_rate: float = 0.0
    latency:             float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert network state to fixed-size float32 vector (dim=16)."""
        vec = np.zeros(16, dtype=np.float32)
        vec[0]  = self.packet_rate / 50_000.0
        vec[1]  = self.anomaly_score
        vec[2]  = self.attack_type / 8.0
        vec[3]  = self.attack_intensity
        vec[4]  = len(self.defense_active) / 8.0
        vec[5]  = self.compromised_nodes / max(self.n_nodes, 1)
        vec[6]  = self.detection_conf
        vec[7]  = self.false_positive_rate
        vec[8]  = min(self.latency / 200.0, 1.0)
        # One-hot attack type in positions 8–15
        if 0 <= self.attack_type < 8:
            vec[8 + self.attack_type] = 1.0
        return vec


class CyberEnvironment:
    """
    Simulated cyber-network environment for the CAESAR framework.

    State space  : 16-dim float vector
    Attack space : 8 attack types × continuous intensity ∈ [0, 1]
    Defense space: 8 discrete actions
    """

    STATE_DIM        = 16
    DEFENSE_DIM      = 8    # defense-state embedding size
    N_ATTACK_TYPES   = len(AttackType)
    N_DEFENSE_ACTIONS= len(DefenseAction)

    def __init__(self, n_nodes: int = 10, seed: int = 42):
        self.n_nodes = n_nodes
        self.rng     = np.random.default_rng(seed)
        self.nodes: List[NetworkNode] = []
        self.state   = NetworkState(n_nodes=n_nodes)
        self.step_count = 0
        self.attack_log: List[Dict] = []
        self.defense_log: List[Dict] = []
        self._build_network()

    # ------------------------------------------------------------------
    def _build_network(self):
        types = (['host'] * 5 + ['server'] * 2 +
                 ['router'] * 2 + ['firewall'] * 1)
        self.nodes = [
            NetworkNode(
                node_id=i,
                node_type=types[i % len(types)],
                vulnerability=self.rng.uniform(0.2, 0.85),
            )
            for i in range(self.n_nodes)
        ]

    def reset(self) -> np.ndarray:
        self.state = NetworkState(n_nodes=self.n_nodes)
        self.step_count = 0
        for n in self.nodes:
            n.is_compromised = False
            n.vulnerability  = self.rng.uniform(0.2, 0.85)
            n.traffic_load   = 0.0
            n.patched        = False
        return self.state.to_vector()

    # ------------------------------------------------------------------
    def inject_attack(self, attack_type: int, intensity: float) -> float:
        """
        Inject attack into the network.
        Returns attack_success ∈ [0, 1].
        """
        attack_type = int(attack_type) % self.N_ATTACK_TYPES
        intensity   = float(np.clip(intensity, 0.0, 1.0))

        self.state.attack_type      = attack_type
        self.state.attack_intensity = intensity

        avg_vuln      = float(np.mean([n.vulnerability for n in self.nodes]))
        defense_block = len(self.state.defense_active) * 0.08

        if attack_type == AttackType.NORMAL:
            self.state.packet_rate   = self.rng.uniform(100, 500)
            self.state.anomaly_score = self.rng.uniform(0.0, 0.15)
            return 0.0

        # Attack-specific dynamics
        params = {
            AttackType.DOS:          (intensity * 8_000,  intensity * 80,  0.80, 0.85),
            AttackType.DDOS:         (intensity * 50_000, intensity * 200, 0.95, 1.10),
            AttackType.PORT_SCAN:    (intensity * 1_500,  0.0,             0.45, 0.30),
            AttackType.BRUTE_FORCE:  (intensity * 800,    0.0,             0.60, 0.65),
            AttackType.DATA_EXFIL:   (intensity * 500,    0.0,             0.40, 0.75),
            AttackType.MITM:         (intensity * 300,    intensity * 20,  0.30, 0.55),
            AttackType.RANSOMWARE:   (intensity * 200,    0.0,             0.75, 1.40),
        }
        pkt, lat, anom_mult, success_mult = params[attack_type]
        self.state.packet_rate   = pkt
        self.state.latency       = lat
        self.state.anomaly_score = min(1.0, intensity * anom_mult)

        raw_success = intensity * avg_vuln * success_mult
        attack_success = float(np.clip(raw_success - defense_block, 0.0, 1.0))

        # Mark nodes as compromised
        n_comp = int(attack_success * self.n_nodes)
        for i in range(n_comp):
            self.nodes[i].is_compromised = True
        self.state.compromised_nodes = n_comp
        self.state.detection_conf    = min(1.0, self.state.anomaly_score +
                                           0.05 * len(self.state.defense_active))

        self.attack_log.append({
            'step': self.step_count, 'type': attack_type,
            'intensity': intensity, 'success': attack_success,
        })
        return attack_success

    # ------------------------------------------------------------------
    def apply_defense(self, action: int) -> Tuple[float, float]:
        """
        Deploy a defense countermeasure.
        Returns (defense_reward, false_positive_cost).
        """
        action = int(action) % self.N_DEFENSE_ACTIONS
        at     = self.state.attack_type
        intens = self.state.attack_intensity

        reward, fpr = 0.0, 0.0

        if action == DefenseAction.NO_ACTION:
            reward = -(intens * 1.5)

        elif action == DefenseAction.BLOCK_IP:
            if at in (AttackType.DOS, AttackType.BRUTE_FORCE, AttackType.DDOS):
                reward = 0.85 * intens
                self.state.compromised_nodes = max(0, self.state.compromised_nodes - 3)
            else:
                reward, fpr = 0.05, 0.25

        elif action == DefenseAction.RATE_LIMIT:
            if at in (AttackType.DOS, AttackType.DDOS):
                reward = 0.70 * intens
                self.state.packet_rate *= 0.25
                self.state.latency     *= 0.40
            else:
                reward, fpr = 0.05, 0.10

        elif action == DefenseAction.ISOLATE_NODE:
            if at in (AttackType.DATA_EXFIL, AttackType.RANSOMWARE, AttackType.MITM):
                reward = 0.90 * intens
                self.state.compromised_nodes = max(0, self.state.compromised_nodes - 5)
            else:
                reward, fpr = 0.15, 0.30

        elif action == DefenseAction.DEPLOY_HONEYPOT:
            if at in (AttackType.PORT_SCAN, AttackType.BRUTE_FORCE):
                reward = 0.65 * intens
                self.state.detection_conf = min(1.0, self.state.detection_conf + 0.25)
            else:
                reward, fpr = 0.10, 0.05

        elif action == DefenseAction.ALERT_ADMIN:
            reward = 0.15
            self.state.detection_conf = min(1.0, self.state.detection_conf + 0.10)

        elif action == DefenseAction.PATCH_VULNERABILITY:
            for n in self.nodes:
                n.vulnerability = max(0.05, n.vulnerability - 0.12)
                n.patched = True
            reward = 0.50

        elif action == DefenseAction.RESET_CONNECTION:
            if at in (AttackType.MITM, AttackType.DDOS):
                reward = 0.55 * intens
                self.state.latency = 0.0
            else:
                reward, fpr = 0.10, 0.15

        # Track active defenses (sliding window of 5)
        if action != DefenseAction.NO_ACTION:
            if action not in self.state.defense_active:
                self.state.defense_active.append(action)
            if len(self.state.defense_active) > 5:
                self.state.defense_active.pop(0)

        self.state.false_positive_rate = fpr
        self.step_count += 1

        self.defense_log.append({
            'step': self.step_count, 'action': action,
            'reward': reward, 'fpr': fpr,
        })

        total = reward - fpr * 0.40
        return float(total), float(fpr)

    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        return self.state.to_vector()

    def get_defense_embedding(self) -> np.ndarray:
        """8-dim defense-state embedding used to condition TA-GAN."""
        vec = np.zeros(self.DEFENSE_DIM, dtype=np.float32)
        vec[0] = len(self.state.defense_active) / 8.0
        vec[1] = self.state.detection_conf
        vec[2] = self.state.false_positive_rate
        vec[3] = self.state.compromised_nodes / max(self.n_nodes, 1)
        for i, a in enumerate(self.state.defense_active[:4]):
            vec[4 + i] = a / 8.0
        return vec

    def network_health(self) -> float:
        """Scalar network health ∈ [0, 1]."""
        avg_vuln    = float(np.mean([n.vulnerability for n in self.nodes]))
        comp_ratio  = self.state.compromised_nodes / max(self.n_nodes, 1)
        lat_factor  = min(1.0, self.state.latency / 200.0)
        h = 1.0 - (0.45 * comp_ratio + 0.30 * avg_vuln + 0.25 * lat_factor)
        return float(np.clip(h, 0.0, 1.0))
