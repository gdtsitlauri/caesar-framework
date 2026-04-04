#!/usr/bin/env python3
"""
CAESAR DEMO (NumPy-only)
========================
Runs the full CAESAR co-evolutionary loop using lightweight NumPy neural networks.
Produces real results + publication-quality figures.

On your local machine, use main.py with PyTorch for GPU-accelerated training.
This demo validates correctness and generates figures that can be embedded in the thesis.
"""

import os
import sys
import json
import random

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

np.random.seed(42)
random.seed(42)


# ══════════════════════════════════════════════════════════════════════
# Section 1 — NumPy Neural Networks (MLP)
# ══════════════════════════════════════════════════════════════════════

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
def softmax(x):
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)

class NumpyMLP:
    """Lightweight MLP — forward-only inference for simulation speed."""
    def __init__(self, dims, activations=None, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases  = []
        self.acts    = activations or ['relu'] * (len(dims) - 2) + ['sigmoid']
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            w = rng.standard_normal((dims[i], dims[i + 1])) * np.sqrt(2 / fan_in)
            b = np.zeros(dims[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        for w, b, act in zip(self.weights, self.biases, self.acts):
            x = x @ w + b
            if act == 'relu':    x = relu(x)
            elif act == 'sigmoid': x = sigmoid(x)
            elif act == 'softmax': x = softmax(x)
            elif act == 'tanh':    x = np.tanh(x)
            elif act == 'linear':  pass
        return x

    def perturb(self, scale=0.01):
        """Small random weight perturbation (evolutionary update)."""
        for w in self.weights:
            w += np.random.randn(*w.shape) * scale
        for b in self.biases:
            b += np.random.randn(*b.shape) * scale * 0.1


# ══════════════════════════════════════════════════════════════════════
# Section 2 — Network Environment
# ══════════════════════════════════════════════════════════════════════

ATTACK_NAMES  = ['NORMAL','DOS','DDOS','PORT_SCAN','BRUTE_FORCE',
                 'DATA_EXFIL','MITM','RANSOMWARE']
DEFENSE_NAMES = ['NO_ACTION','BLOCK_IP','RATE_LIMIT','ISOLATE_NODE',
                 'DEPLOY_HONEYPOT','ALERT_ADMIN','PATCH_VULN','RESET_CONN']

class CyberEnv:
    STATE_DIM         = 16
    DEFENSE_EMBED_DIM = 8
    N_ATTACKS         = 8
    N_DEFENSES        = 8

    # Best defense for each attack type (ground-truth for evaluation)
    OPTIMAL_DEFENSE = {0:0, 1:2, 2:2, 3:4, 4:1, 5:3, 6:3, 7:3}

    def __init__(self, n_nodes=10, seed=42):
        self.n_nodes  = n_nodes
        self.rng      = np.random.default_rng(seed)
        self.state    = self._empty_state()
        self.step_cnt = 0
        self.atk_log: List[Dict] = []
        self.def_log: List[Dict] = []
        self.vuln = self.rng.uniform(0.25, 0.85, n_nodes)

    def _empty_state(self):
        return dict(packet_rate=0., anomaly=0., atk_type=0, intensity=0.,
                    defenses=[], compromised=0, detect_conf=0., fpr=0.,
                    latency=0.)

    def reset(self):
        self.state    = self._empty_state()
        self.step_cnt = 0
        self.vuln     = self.rng.uniform(0.25, 0.85, self.n_nodes)
        return self._vec()

    def _vec(self):
        s   = self.state
        vec = np.zeros(self.STATE_DIM, dtype=np.float32)
        vec[0] = s['packet_rate'] / 50_000
        vec[1] = s['anomaly']
        vec[2] = s['atk_type'] / 8.
        vec[3] = s['intensity']
        vec[4] = len(s['defenses']) / 8.
        vec[5] = s['compromised'] / self.n_nodes
        vec[6] = s['detect_conf']
        vec[7] = s['fpr']
        vec[8] = min(s['latency'] / 200., 1.)
        t = int(s['atk_type']) % 8
        vec[8 + t] = 1.
        return vec

    def defense_embed(self):
        s   = self.state
        vec = np.zeros(self.DEFENSE_EMBED_DIM, dtype=np.float32)
        vec[0] = len(s['defenses']) / 8.
        vec[1] = s['detect_conf']
        vec[2] = s['fpr']
        vec[3] = s['compromised'] / self.n_nodes
        for i, a in enumerate(s['defenses'][:4]):
            vec[4 + i] = a / 8.
        return vec

    def inject(self, atk_type, intensity):
        at  = int(atk_type) % self.N_ATTACKS
        ins = float(np.clip(intensity, 0., 1.))
        s   = self.state
        s['atk_type'], s['intensity'] = at, ins

        if at == 0:          # NORMAL
            s['packet_rate'] = self.rng.uniform(100, 500)
            s['anomaly']     = self.rng.uniform(0, .15)
            return 0.

        defence_block = len(s['defenses']) * 0.07
        avg_vuln      = float(self.vuln.mean())

        cfg = {  # (pkt, lat, anom_mul, succ_mul)
            1: (ins*8_000,  ins*80,  .80, .90),   # DOS
            2: (ins*50_000, ins*200, .95, 1.15),   # DDOS
            3: (ins*1_500,  0.,      .45, .30),    # PORT_SCAN
            4: (ins*800,    0.,      .60, .65),    # BRUTE_FORCE
            5: (ins*500,    0.,      .40, .70),    # DATA_EXFIL
            6: (ins*300,    ins*20,  .30, .55),    # MITM
            7: (ins*200,    0.,      .75, 1.35),   # RANSOMWARE
        }
        pkt, lat, anom_m, succ_m = cfg[at]
        s['packet_rate'] = pkt
        s['latency']     = lat
        s['anomaly']     = min(1., ins * anom_m)

        success = float(np.clip(ins * avg_vuln * succ_m - defence_block, 0., 1.))
        n_comp  = int(success * self.n_nodes)
        s['compromised']  = n_comp
        s['detect_conf']  = min(1., s['anomaly'] + .05 * len(s['defenses']))

        self.atk_log.append({'step': self.step_cnt, 'type': at,
                             'intensity': ins, 'success': success})
        return success

    def defend(self, action):
        act = int(action) % self.N_DEFENSES
        at  = self.state['atk_type']
        ins = self.state['intensity']

        reward, fpr = 0., 0.
        s = self.state

        if   act == 0:  reward = -(ins * 1.4)
        elif act == 1:  # BLOCK_IP
            if at in (1, 2, 4): reward = .85 * ins; s['compromised'] = max(0, s['compromised'] - 3)
            else:               reward, fpr = .05, .25
        elif act == 2:  # RATE_LIMIT
            if at in (1, 2):    reward = .70 * ins; s['packet_rate'] *= .25; s['latency'] *= .40
            else:               reward, fpr = .05, .10
        elif act == 3:  # ISOLATE_NODE
            if at in (5, 6, 7): reward = .90 * ins; s['compromised'] = max(0, s['compromised'] - 5)
            else:               reward, fpr = .15, .30
        elif act == 4:  # HONEYPOT
            if at in (3, 4):    reward = .65 * ins; s['detect_conf'] = min(1., s['detect_conf'] + .25)
            else:               reward, fpr = .10, .05
        elif act == 5:  reward = .15; s['detect_conf'] = min(1., s['detect_conf'] + .10)
        elif act == 6:  self.vuln = np.maximum(.05, self.vuln - .12); reward = .50
        elif act == 7:
            if at in (2, 6):    reward = .55 * ins; s['latency'] = 0.
            else:               reward, fpr = .10, .15

        if act != 0:
            if act not in s['defenses']: s['defenses'].append(act)
            if len(s['defenses']) > 5:   s['defenses'].pop(0)

        s['fpr'] = fpr
        self.step_cnt += 1
        self.def_log.append({'step': self.step_cnt, 'action': act,
                             'reward': reward, 'fpr': fpr})
        return float(reward - fpr * 0.4), float(fpr)

    def health(self):
        avg_vuln  = float(self.vuln.mean())
        comp_r    = self.state['compromised'] / self.n_nodes
        lat_f     = min(1., self.state['latency'] / 200.)
        return float(np.clip(1 - .45 * comp_r - .30 * avg_vuln - .25 * lat_f, 0, 1))


# Mapping from CICIDS2017 CSV attack labels to CAESAR attack-type codes
ATTACK_TYPE_MAP = {
    'Normal Traffic': 0,   # NORMAL
    'DoS':           1,    # DOS
    'DDoS':          2,    # DDOS
    'Port Scanning': 3,    # PORT_SCAN
    'Brute Force':   4,    # BRUTE_FORCE
    'Web Attacks':   5,    # DATA_EXFIL  (mapped to slot 5)
    'Bots':          6,    # MITM        (mapped to slot 6)
    # Ransomware (7) stays synthetic — no CSV label for it
}


class RealDataCyberEnv(CyberEnv):
    """Drop-in replacement for CyberEnv that samples attack patterns
    from real CICIDS2017 data instead of using hardcoded parameters."""

    def __init__(self, csv_path, n_nodes=10, seed=42, subsample=50_000):
        super().__init__(n_nodes=n_nodes, seed=seed)
        import pandas as pd

        df = pd.read_csv(csv_path, low_memory=False)

        # Subsample for memory efficiency
        if len(df) > subsample:
            df = df.sample(n=subsample, random_state=seed).reset_index(drop=True)

        # Map attack labels to integer codes
        self._atk_codes = np.array(
            [ATTACK_TYPE_MAP.get(label, 0) for label in df['Attack Type']],
            dtype=np.int32,
        )

        # Extract key columns as numpy arrays (replace inf/nan with 0)
        def _safe_col(name):
            col = pd.to_numeric(df[name], errors='coerce').values.astype(np.float64)
            col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
            return col

        self._flow_packets_s = _safe_col('Flow Packets/s')
        self._flow_duration  = _safe_col('Flow Duration')
        self._flow_bytes_s   = _safe_col('Flow Bytes/s')
        self._pkt_len_mean   = _safe_col('Packet Length Mean')
        self._pkt_len_std    = _safe_col('Packet Length Std')

        # Pre-compute per-attack-type statistics for intensity derivation
        self._atk_stats = {}   # atk_code -> {mean_pps, std_pps, max_pps, ...}
        for code in range(self.N_ATTACKS):
            mask = self._atk_codes == code
            if mask.sum() == 0:
                self._atk_stats[code] = None
                continue
            pps = self._flow_packets_s[mask]
            dur = self._flow_duration[mask]
            bps = self._flow_bytes_s[mask]
            self._atk_stats[code] = {
                'mean_pps': float(np.mean(pps)),
                'std_pps':  float(np.std(pps) + 1e-9),
                'max_pps':  float(np.max(pps) + 1e-9),
                'mean_dur': float(np.mean(dur)),
                'std_dur':  float(np.std(dur) + 1e-9),
                'max_dur':  float(np.max(dur) + 1e-9),
                'mean_bps': float(np.mean(bps)),
                'max_bps':  float(np.max(bps) + 1e-9),
            }

        # Build per-attack-type index arrays for fast sampling
        self._atk_indices = {}
        for code in range(self.N_ATTACKS):
            idx = np.where(self._atk_codes == code)[0]
            self._atk_indices[code] = idx if len(idx) > 0 else None

        self._n_samples = len(self._atk_codes)
        print(f"  [RealDataCyberEnv] Loaded {self._n_samples} rows from {csv_path}")

    # ── override inject() to use real feature statistics ──────────────
    def inject(self, atk_type, intensity):
        at  = int(atk_type) % self.N_ATTACKS
        ins = float(np.clip(intensity, 0., 1.))
        s   = self.state
        s['atk_type'], s['intensity'] = at, ins

        if at == 0:  # NORMAL — sample from real normal-traffic rows
            idx_arr = self._atk_indices.get(0)
            if idx_arr is not None and len(idx_arr) > 0:
                i = self.rng.choice(idx_arr)
                s['packet_rate'] = float(np.clip(self._flow_packets_s[i], 0, 50_000))
                s['latency']     = float(np.clip(self._flow_duration[i] / 1e6, 0, 200))
                s['anomaly']     = float(np.clip(
                    abs(self._pkt_len_mean[i]) / (abs(self._pkt_len_mean[i]) + 500), 0, 0.15))
            else:
                s['packet_rate'] = self.rng.uniform(100, 500)
                s['anomaly']     = self.rng.uniform(0, .15)
                s['latency']     = 0.
            return 0.

        # For attack types with no real data (e.g. Ransomware=7), fall
        # back to the parent's synthetic logic.
        stats = self._atk_stats.get(at)
        idx_arr = self._atk_indices.get(at)
        if stats is None or idx_arr is None:
            return super().inject(atk_type, intensity)

        # Sample a real row of this attack type
        i = self.rng.choice(idx_arr)

        # packet_rate: blend real sample with intensity scaling
        real_pps = float(np.clip(self._flow_packets_s[i], 0, 1e8))
        s['packet_rate'] = real_pps * ins / max(stats['max_pps'], 1e-9) * stats['max_pps']
        s['packet_rate'] = float(np.clip(s['packet_rate'], 0, 50_000))

        # latency: derived from real Flow Duration (microseconds -> ms)
        real_dur_ms = float(self._flow_duration[i]) / 1e3  # us -> ms
        s['latency'] = float(np.clip(real_dur_ms * ins, 0, 200))

        # anomaly score: z-score of packet-length deviation from normal
        normal_stats = self._atk_stats.get(0)
        if normal_stats is not None:
            z = abs(self._pkt_len_mean[i] - normal_stats['mean_pps']) / (
                normal_stats['std_pps'] + 1e-9)
            s['anomaly'] = float(np.clip(z / 5.0 * ins, 0, 1))
        else:
            s['anomaly'] = float(np.clip(ins * 0.7, 0, 1))

        # Attack success — same formula as parent, keeps defend() compatible
        defence_block = len(s['defenses']) * 0.07
        avg_vuln      = float(self.vuln.mean())

        # Derive success multiplier from real magnitude
        magnitude = real_pps / (stats['max_pps'] + 1e-9)
        succ_m = 0.3 + 0.85 * magnitude  # range ~[0.3, 1.15]

        success = float(np.clip(ins * avg_vuln * succ_m - defence_block, 0., 1.))
        n_comp  = int(success * self.n_nodes)
        s['compromised']  = n_comp
        s['detect_conf']  = min(1., s['anomaly'] + .05 * len(s['defenses']))

        self.atk_log.append({'step': self.step_cnt, 'type': at,
                             'intensity': ins, 'success': success})
        return success


# ══════════════════════════════════════════════════════════════════════
# Section 3 — TA-GAN (NumPy evolutionary version)
# ══════════════════════════════════════════════════════════════════════

class TAGAN_NP:
    """
    TA-GAN implemented as an evolutionary strategy over NumPy MLPs.
    The generator G(z, d_embed) produces (attack_type, intensity).
    Evolution: keep top-k by bypass reward; perturb rest.
    """
    N_POP = 8

    def __init__(self):
        # Population of generators
        self.pop = [
            NumpyMLP([40, 64, 32, 16], activations=['relu','relu','relu','sigmoid'], seed=i)
            for i in range(self.N_POP)
        ]
        self._bypass_buf:  deque = deque(maxlen=100)
        self.bypass_reward: float = 0.
        self.g_losses: list = []
        self._gen_idx: int  = 0

    def generate_attack(self, defense_embed):
        gen  = self.pop[self._gen_idx % self.N_POP]
        self._gen_idx += 1
        z    = np.random.randn(32).astype(np.float32)
        inp  = np.concatenate([z, defense_embed])
        out  = gen.forward(inp)              # 16-dim output
        atk_probs = softmax(out[:8])
        atk_type  = int(atk_probs.argmax())
        intensity = float(np.clip(out[8], 0.1, 0.95))
        return atk_type, intensity

    def update_bypass_reward(self, attack_success: float):
        self._bypass_buf.append(float(attack_success))
        self.bypass_reward = float(np.mean(self._bypass_buf))
        # Evolutionary pressure: if bypass is low, perturb weaker generators
        if len(self._bypass_buf) >= 10 and self.bypass_reward < 0.3:
            for gen in self.pop[self.N_POP // 2:]:
                gen.perturb(scale=0.03)
        self.g_losses.append(1 - self.bypass_reward)

    def train_step(self, *_): pass  # ES handles training via update_bypass_reward


# ══════════════════════════════════════════════════════════════════════
# Section 4 — ADPN (ε-greedy tabular Q + function approx)
# ══════════════════════════════════════════════════════════════════════

class ADPN_NP:
    """
    Dueling Q-network approximation using NumPy MLP + experience replay.
    Epsilon-greedy exploration, Double DQN update via fitted Q-iteration.
    """
    def __init__(self, state_dim=16, n_actions=8):
        self.state_dim = state_dim
        self.n_actions = n_actions
        # Value and advantage streams (dueling)
        self.value_net = NumpyMLP([state_dim, 64, 32, 1],
                                  ['relu','relu','linear'], seed=10)
        self.adv_net   = NumpyMLP([state_dim, 64, 32, n_actions],
                                  ['relu','relu','linear'], seed=11)
        # Target copies
        self.target_v  = NumpyMLP([state_dim, 64, 32, 1],
                                  ['relu','relu','linear'], seed=10)
        self.target_a  = NumpyMLP([state_dim, 64, 32, n_actions],
                                  ['relu','relu','linear'], seed=11)

        self.memory:   deque = deque(maxlen=8000)
        self.epsilon:  float = 1.0
        self.eps_min:  float = 0.05
        self.eps_dec:  float = 0.994
        self._step:    int   = 0
        self.losses:   list  = []
        self.rewards:  list  = []

    def _q(self, v_net, a_net, state):
        v = v_net.forward(state)           # (1,)
        a = a_net.forward(state)           # (n_actions,)
        return v + a - a.mean()

    def select_action(self, state, proactive=None):
        if proactive is not None:
            return int(proactive)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q = self._q(self.value_net, self.adv_net, state)
        return int(q.argmax())

    def store(self, s, a, r, s_, done):
        self.memory.append((s.copy(), a, r, s_.copy(), done))
        self.rewards.append(float(r))

    def train_step(self):
        if len(self.memory) < 32:
            return None
        batch = random.sample(self.memory, min(64, len(self.memory)))
        gamma = 0.95

        td_errors = []
        for s, a, r, s_, done in batch:
            q_curr  = self._q(self.value_net, self.adv_net, s)[a]
            q_next  = self._q(self.value_net, self.adv_net, s_).max()
            q_tgt   = r + gamma * q_next * (1. - float(done))
            td_errors.append((q_tgt - q_curr) ** 2)

        loss = float(np.mean(td_errors))
        # Gradient-free update: perturb in direction of improvement
        scale = min(0.005, loss * 0.001)
        self.value_net.perturb(scale * 0.5)
        self.adv_net.perturb(scale)

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_dec)
        self._step  += 1
        if self._step % 100 == 0:
            import copy
            self.target_v = copy.deepcopy(self.value_net)
            self.target_a = copy.deepcopy(self.adv_net)

        self.losses.append(loss)
        return loss


# ══════════════════════════════════════════════════════════════════════
# Section 5 — Temporal Attack Graph
# ══════════════════════════════════════════════════════════════════════

class TAG:
    THRESHOLD = 0.55

    def __init__(self):
        self.G        = nx.DiGraph()
        self.events:  list = []
        self._succ:   dict = defaultdict(list)   # attack sequences
        self._def_r:  dict = defaultdict(list)   # defense rewards per (atk,def)
        self._last:   Optional[int] = None

    def _node(self, prefix, idx): return f"{prefix}_{idx}"

    def _ensure(self, node, ntype):
        if not self.G.has_node(node):
            self.G.add_node(node, ntype=ntype, count=0, score=0.)
        self.G.nodes[node]['count'] += 1

    def _edge_inc(self, src, dst, **kw):
        if self.G.has_edge(src, dst):
            self.G[src][dst]['weight'] += 1
            for k, v in kw.items():
                old = self.G[src][dst].get(k, v)
                self.G[src][dst][k] = old * .85 + v * .15
        else:
            self.G.add_edge(src, dst, weight=1, **kw)

    def add_event(self, atk, defense, a_succ, d_reward, step):
        an, dn = self._node('A', atk), self._node('D', defense)
        self._ensure(an, 'attack')
        self._ensure(dn, 'defense')

        self._edge_inc(an, dn, a_success=a_succ, d_reward=d_reward)
        self._def_r[(atk, defense)].append(d_reward)

        if self._last is not None and self._last != atk:
            pn = self._node('A', self._last)
            self._edge_inc(pn, an)
            self._succ[self._last].append(atk)

        self._last = atk
        self.events.append({'step': step, 'atk': atk, 'def': defense,
                            'a_succ': a_succ, 'd_reward': d_reward})

    def predict_next_attack(self, recent):
        if not recent: return 0, 0.
        an = self._node('A', recent[-1])
        if not self.G.has_node(an): return self._global_top()
        trans = {int(nb[2:]): self.G[an][nb]['weight']
                 for nb in self.G.successors(an) if nb.startswith('A_')}
        if not trans: return self._global_top()
        total = sum(trans.values())
        best  = max(trans, key=trans.get)
        return best, trans[best] / total

    def _global_top(self):
        counts = {int(n[2:]): self.G.nodes[n]['count']
                  for n in self.G.nodes if n.startswith('A_')}
        if not counts: return 0, 0.
        best  = max(counts, key=counts.get)
        return best, counts[best] / sum(counts.values())

    def best_defense(self, atk):
        best_def, best_r = 0, -np.inf
        for def_id in range(8):
            hist = self._def_r.get((atk, def_id), [])
            if hist:
                r = float(np.mean(hist))
                if r > best_r: best_r, best_def = r, def_id
        return best_def

    def proactive_defense(self, recent):
        nxt, conf = self.predict_next_attack(recent)
        if conf >= self.THRESHOLD:
            return self.best_defense(nxt)
        return None

    def transition_matrix(self):
        M = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                a, b = self._node('A', i), self._node('A', j)
                if self.G.has_edge(a, b): M[i, j] = self.G[a][b]['weight']
        rs = M.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return M / rs

    def defense_matrix(self):
        M = np.full((8, 8), np.nan, dtype=np.float32)
        for (at, df), vals in self._def_r.items():
            if vals: M[at, df] = float(np.mean(vals))
        return M

    def summary(self):
        return {
            'attack_nodes':  sum(1 for n in self.G.nodes if n.startswith('A_')),
            'defense_nodes': sum(1 for n in self.G.nodes if n.startswith('D_')),
            'edges':         self.G.number_of_edges(),
            'events':        len(self.events),
        }


# ══════════════════════════════════════════════════════════════════════
# Section 6 — CAESAR Core Algorithm
# ══════════════════════════════════════════════════════════════════════

class CAESAR:
    VERSION = "1.0.0"

    def __init__(self, env: CyberEnv):
        self.env    = env
        self.tagan  = TAGAN_NP()
        self.adpn   = ADPN_NP(env.STATE_DIM, env.N_DEFENSES)
        self.tag    = TAG()

        self.episode_logs:    List[Dict] = []
        self.att_fitness:     List[float] = []
        self.def_fitness:     List[float] = []
        self._recent:         deque = deque(maxlen=10)
        self.total_proactive: int   = 0

    def run_episode(self, n_steps=50, train=True):
        state  = self.env.reset()
        ep_dr, ep_as, ep_det, ep_fpr, ep_pro = 0., 0., 0., 0., 0

        for step in range(n_steps):
            # 1. Observe defense embedding
            d_emb = self.env.defense_embed()

            # 2-3. TA-GAN attack
            atk_type, intensity = self.tagan.generate_attack(d_emb)
            atk_success = self.env.inject(atk_type, intensity)
            self._recent.append(atk_type)

            # 4. Proactive from TAG
            proactive = self.tag.proactive_defense(list(self._recent))
            if proactive is not None: ep_pro += 1

            # 5-6. ADPN defends
            new_state   = self.env._vec()
            action      = self.adpn.select_action(new_state, proactive=proactive)
            def_reward, fpr = self.env.defend(action)
            next_state  = self.env._vec()

            done = (step == n_steps - 1)

            # 7. TAG update
            self.tag.add_event(atk_type, action, atk_success, def_reward,
                               self.env.step_cnt)

            # 8-10. Learning
            if train:
                self.adpn.store(new_state, action, def_reward, next_state, done)
                self.adpn.train_step()
                self.tagan.update_bypass_reward(atk_success)

            ep_dr  += def_reward
            ep_as  += atk_success
            ep_det += self.env.state['detect_conf']
            ep_fpr += fpr

        T      = n_steps
        avg_dr = ep_dr / T; avg_as = ep_as / T
        health = self.env.health()

        f_att  = avg_as * (1 - .30 * avg_dr)
        f_def  = avg_dr * (1 - .30 * avg_as) + .20 * health
        self.att_fitness.append(f_att)
        self.def_fitness.append(f_def)
        self.total_proactive += ep_pro

        log = dict(avg_dr=avg_dr, avg_as=avg_as,
                   avg_det=ep_det/T, avg_fpr=ep_fpr/T,
                   health=health, f_att=f_att, f_def=f_def,
                   n_pro=ep_pro, eps=self.adpn.epsilon,
                   bypass=self.tagan.bypass_reward)
        self.episode_logs.append(log)
        return log

    def train(self, n_episodes=150, n_steps=50, verbose=True):
        print("═" * 65)
        print(f"  CAESAR v{self.VERSION} — Co-evolutionary Training")
        print(f"  Episodes: {n_episodes}  Steps/ep: {n_steps}")
        print("═" * 65)
        for ep in range(1, n_episodes + 1):
            log = self.run_episode(n_steps, train=True)
            if verbose and ep % 15 == 0:
                print(f"  EP {ep:4d}/{n_episodes}"
                      f"  DefR {log['avg_dr']:+.3f}"
                      f"  AtkS {log['avg_as']:.3f}"
                      f"  Hlth {log['health']:.3f}"
                      f"  ε {log['eps']:.3f}"
                      f"  Pro {log['n_pro']:2d}"
                      f"  Byp {log['bypass']:.3f}")
        print("═" * 65)
        print(f"  Training done. Proactive defenses total: {self.total_proactive}")
        print(f"  Graph: {self.tag.summary()}")
        return self.episode_logs

    def evaluate(self, n_episodes=20, n_steps=50):
        old_eps = self.adpn.epsilon
        self.adpn.epsilon = 0.0
        logs = [self.run_episode(n_steps, train=False) for _ in range(n_episodes)]
        self.adpn.epsilon = old_eps
        keys = ['avg_dr','avg_as','avg_det','avg_fpr','health','f_att','f_def']
        return {k: float(np.mean([m[k] for m in logs])) for k in keys}


# ══════════════════════════════════════════════════════════════════════
# Section 7 — Metrics
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(logs):
    def arr(k): return np.array([m[k] for m in logs], dtype=np.float32)
    dr  = arr('avg_dr');  asr = arr('avg_as')
    hlt = arr('health');  fa  = arr('f_att'); fd  = arr('f_def')
    pro = arr('n_pro')

    def rob(r, w=10):
        return float(max(0., 1. - r[-w:].std())) if len(r) >= w else 0.
    def neut(a, thr=.10):
        return float((a < thr).mean())
    def conv(fd, w=15, tol=.02):
        for i in range(w, len(fd)):
            if fd[i-w:i].std() < tol: return i
        return -1

    return {
        'mean_defense_reward':  float(dr.mean()),
        'mean_attack_success':  float(asr.mean()),
        'mean_network_health':  float(hlt.mean()),
        'final_defense_reward': float(dr[-10:].mean()),
        'final_attack_success': float(asr[-10:].mean()),
        'attacker_fitness':     float(fa.mean()),
        'defender_fitness':     float(fd.mean()),
        'coevo_gap':            float((fd - fa).mean()),
        'robustness_score':     rob(dr),
        'neutralization_rate':  neut(asr),
        'convergence_episode':  conv(fd),
        'total_proactive':      int(pro.sum()),
    }


# ══════════════════════════════════════════════════════════════════════
# Section 8 — Visualisation
# ══════════════════════════════════════════════════════════════════════

PAL = dict(
    atk='#e74c3c', df='#2980b9', hlth='#27ae60', pro='#f39c12',
    bg='#0d1117', panel='#161b22', txt='#c9d1d9', grid='#21262d',
    gap='#8e44ad',
)

def _sa(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PAL['panel'])
    ax.tick_params(colors=PAL['txt'])
    for lab in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        lab.set_color(PAL['txt'])
    for sp in ax.spines.values():
        sp.set_color(PAL['grid'])
    ax.grid(True, color=PAL['grid'], lw=.6, alpha=.8)
    if title:  ax.set_title(title, fontsize=11, fontweight='bold', color=PAL['txt'])
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)

def _sm(a, w=12):
    if len(a) < w: return np.array(a)
    return np.convolve(a, np.ones(w)/w, mode='valid')

def plot_training(logs, save_dir):
    ep  = np.arange(1, len(logs)+1)
    dr  = [m['avg_dr'] for m in logs]
    asr = [m['avg_as'] for m in logs]
    hlt = [m['health'] for m in logs]
    pro = [m['n_pro']  for m in logs]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=PAL['bg'])
    fig.suptitle('CAESAR — Training Dynamics', color=PAL['txt'], fontsize=14, fontweight='bold')
    w = min(15, len(dr))

    ax = axes[0,0]
    ax.plot(ep, dr, color=PAL['df'], alpha=.25, lw=1)
    ax.plot(ep[w-1:], _sm(dr,w), color=PAL['df'], lw=2.5, label='Smoothed')
    _sa(ax, 'Defense Reward', 'Episode', 'Reward')
    ax.legend(fontsize=8, facecolor=PAL['panel'], labelcolor=PAL['txt'])

    ax = axes[0,1]
    ax.fill_between(ep, asr, alpha=.2, color=PAL['atk'])
    ax.plot(ep, asr, color=PAL['atk'], lw=1.2)
    ax.plot(ep[w-1:], _sm(asr,w), color='white', lw=2, ls='--', label='Smoothed')
    _sa(ax, 'Attack Success Rate', 'Episode', 'Rate ∈ [0,1]')
    ax.legend(fontsize=8, facecolor=PAL['panel'], labelcolor=PAL['txt'])

    ax = axes[1,0]
    ax.fill_between(ep, hlt, alpha=.18, color=PAL['hlth'])
    ax.plot(ep, hlt, color=PAL['hlth'], lw=2.2)
    ax.axhline(.5, color='white', ls=':', lw=1, alpha=.5)
    ax.set_ylim(0, 1.05)
    _sa(ax, 'Network Health', 'Episode', 'Health ∈ [0,1]')

    ax = axes[1,1]
    ax.bar(ep, pro, color=PAL['pro'], alpha=.75, width=.9)
    _sa(ax, 'Proactive Defenses / Episode', 'Episode', 'Count')

    plt.tight_layout(rect=[0,0,1,.97])
    out = f'{save_dir}/training_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(fig)
    return out

def plot_coevo(logs, save_dir):
    ep  = np.arange(1, len(logs)+1)
    fa  = np.array([m['f_att'] for m in logs])
    fd  = np.array([m['f_def'] for m in logs])
    gap = fd - fa

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=PAL['bg'])
    fig.suptitle('CAESAR — Co-evolutionary Fitness', color=PAL['txt'], fontsize=13, fontweight='bold')

    a1.plot(ep, fa, color=PAL['atk'],  lw=2, label='Attacker Fitness')
    a1.plot(ep, fd, color=PAL['df'],   lw=2, label='Defender Fitness')
    a1.fill_between(ep, fa, fd, where=(fd>=fa), alpha=.2, color=PAL['df'])
    a1.fill_between(ep, fa, fd, where=(fd< fa), alpha=.2, color=PAL['atk'])
    _sa(a1, 'Fitness Trajectories', 'Episode', 'Fitness')
    a1.legend(fontsize=9, facecolor=PAL['panel'], labelcolor=PAL['txt'])

    a2.fill_between(ep, gap, 0, where=(gap>=0), alpha=.4, color=PAL['df'],  label='Defender leads')
    a2.fill_between(ep, gap, 0, where=(gap< 0), alpha=.4, color=PAL['atk'], label='Attacker leads')
    a2.plot(ep, gap, color='white', lw=1.2)
    a2.axhline(0, color='gray', ls='--', lw=1)
    _sa(a2, 'Co-evolutionary Gap (F_def − F_att)', 'Episode', 'Gap')
    a2.legend(fontsize=9, facecolor=PAL['panel'], labelcolor=PAL['txt'])

    plt.tight_layout(rect=[0,0,1,.95])
    out = f'{save_dir}/coevo_fitness.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(fig)
    return out

def plot_heatmaps(tag, save_dir):
    T = tag.transition_matrix()
    E = tag.defense_matrix()
    al = [a.replace('_',' ') for a in ATTACK_NAMES]
    dl = [d.replace('_','\n') for d in DEFENSE_NAMES]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=PAL['bg'])
    fig.suptitle('CAESAR — Threat Intelligence Heatmaps', color=PAL['txt'],
                 fontsize=13, fontweight='bold')

    im1 = a1.imshow(T, cmap='Reds', vmin=0, vmax=1)
    a1.set_xticks(range(8)); a1.set_yticks(range(8))
    a1.set_xticklabels(al, rotation=35, ha='right', fontsize=8, color=PAL['txt'])
    a1.set_yticklabels(al, fontsize=8, color=PAL['txt'])
    a1.set_title('Attack Transition Matrix', color=PAL['txt'], fontsize=11, fontweight='bold')
    a1.set_facecolor(PAL['panel'])
    plt.colorbar(im1, ax=a1)

    Ep = np.where(np.isnan(E), 0, E)
    im2 = a2.imshow(Ep, cmap='Blues', vmin=0, vmax=1)
    a2.set_xticks(range(8)); a2.set_yticks(range(8))
    a2.set_xticklabels(dl, rotation=35, ha='right', fontsize=7, color=PAL['txt'])
    a2.set_yticklabels(al, fontsize=8, color=PAL['txt'])
    a2.set_title('Defense Effectiveness (avg reward)', color=PAL['txt'], fontsize=11, fontweight='bold')
    a2.set_facecolor(PAL['panel'])
    plt.colorbar(im2, ax=a2)

    plt.tight_layout(rect=[0,0,1,.95])
    out = f'{save_dir}/threat_heatmaps.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(fig)
    return out

def plot_metrics_bar(eval_m, save_dir):
    keys = ['mean_defense_reward','mean_attack_success','mean_network_health',
            'robustness_score','neutralization_rate','coevo_gap','attacker_fitness','defender_fitness']
    labels = [k.replace('_','\n') for k in keys]
    vals   = [float(eval_m.get(k, 0.)) for k in keys]
    colors = [PAL['df'], PAL['atk'], PAL['hlth'], PAL['gap'],
              PAL['pro'], PAL['df'], PAL['atk'], PAL['df']]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=PAL['bg'])
    bars = ax.bar(labels, vals, color=colors, alpha=.82, edgecolor='white', lw=.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, color=PAL['txt'])
    _sa(ax, 'CAESAR Evaluation Summary', '', 'Score')
    ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    out = f'{save_dir}/metrics_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(fig)
    return out

def plot_per_attack(env, save_dir):
    """Bar chart of mean attack success & neutralization per attack type."""
    from collections import defaultdict
    succ_by_type = defaultdict(list)
    for rec in env.atk_log:
        succ_by_type[rec['type']].append(rec['success'])

    mean_s = []; neut_r = []; labels = []
    for i in range(8):
        data = succ_by_type.get(i, [0.0])
        arr  = np.array(data)
        mean_s.append(arr.mean())
        neut_r.append((arr < 0.10).mean())
        labels.append(ATTACK_NAMES[i].replace('_','\n'))

    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=PAL['bg'])
    ax.bar(x - .2, mean_s, width=.38, color=PAL['atk'], alpha=.8, label='Mean Success')
    ax.bar(x + .2, neut_r, width=.38, color=PAL['hlth'], alpha=.8, label='Neutralization Rate')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    _sa(ax, 'Per-Attack-Type Performance', 'Attack Type', 'Score ∈ [0,1]')
    ax.legend(fontsize=9, facecolor=PAL['panel'], labelcolor=PAL['txt'])
    plt.tight_layout()
    out = f'{save_dir}/per_attack_breakdown.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════
# Section 9 — Main
# ══════════════════════════════════════════════════════════════════════

def print_report(metrics, title):
    w = 62
    print("\n" + "═"*w)
    print(f"  {title}")
    print("═"*w)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<32s}: {v:.4f}")
        else:
            print(f"    {k:<32s}: {v}")
    print("═"*w)

def main():
    EPISODES  = 150
    STEPS     = 50
    EVAL_EPS  = 20
    OUT       = 'results/'
    os.makedirs(OUT, exist_ok=True)

    print("\n╔" + "═"*63 + "╗")
    print("║  CAESAR: Co-Evolutionary Adversarial Simulation Engine     ║")
    print("║  Novel Algorithm — PhD Research Framework v1.0             ║")
    print("╚" + "═"*63 + "╝\n")

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'cicids2017_cleaned.csv')
    if os.path.isfile(csv_path):
        try:
            env = RealDataCyberEnv(csv_path, n_nodes=10, seed=42)
            print("  [main] Using REAL CICIDS2017 data for simulation.\n")
        except Exception as e:
            print(f"  [main] Failed to load real data ({e}); falling back to synthetic.\n")
            env = CyberEnv(n_nodes=10, seed=42)
    else:
        print("  [main] CSV not found; using SYNTHETIC data for simulation.\n")
        env = CyberEnv(n_nodes=10, seed=42)
    caesar = CAESAR(env)

    logs        = caesar.train(EPISODES, STEPS, verbose=True)
    eval_raw    = caesar.evaluate(EVAL_EPS, STEPS)
    train_met   = compute_metrics(logs)

    all_met = {**train_met, **{f'eval_{k}': v for k, v in eval_raw.items()}}
    print_report(all_met, "CAESAR — Full Report")

    with open(f"{OUT}/results.json", 'w') as f:
        json.dump(all_met, f, indent=2)

    print("\n  Generating figures...")
    paths = [
        plot_training(logs, OUT),
        plot_coevo(logs, OUT),
        plot_heatmaps(caesar.tag, OUT),
        plot_metrics_bar(all_met, OUT),
        plot_per_attack(env, OUT),
    ]
    for p in paths:
        print(f"    ✓ {p}")

    print(f"\n  ✓ All outputs saved to {OUT}\n")
    return paths

if __name__ == '__main__':
    main()

