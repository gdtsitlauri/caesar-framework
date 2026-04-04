"""
Temporal Attack Graph (TAG)
============================
Novel Contribution #3 of the CAESAR framework.

The TAG is a directed graph whose nodes are attack types and defense actions,
and whose edges carry temporal co-occurrence statistics accumulated across
the entire training run.

Key capabilities
────────────────
1. predict_next_attack()   — Markov-chain-style attack sequence prediction
2. get_best_defense()      — historically most effective defense per attack
3. get_proactive_defense() — combines 1+2 for confidence-gated preemption
4. summary()               — human-readable stats for reporting

Why a graph?
• Attacks rarely occur in isolation; they form sequences (recon → exploit → exfil).
• A graph naturally represents transition probabilities between attack phases.
• PageRank scores reveal "central" attack nodes worth defending proactively.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


PROACTIVE_THRESHOLD = 0.55   # confidence needed to trigger proactive defense


class TemporalAttackGraph:
    """
    Directed, weighted, temporal knowledge graph for attack-defense patterns.

    Node types
    ──────────
    • 'A_k'  : attack type k  (k ∈ 0..7)
    • 'D_j'  : defense action j (j ∈ 0..7)

    Edge semantics
    ──────────────
    • A_k → A_k' : attack k followed by attack k' (sequence transition)
    • A_k → D_j  : attack k was met with defense j
    • D_j → A_k' : after defense j, attacker chose k' next
    """

    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self.events: deque = deque(maxlen=2000)

        # Per-entity accumulators
        self._atk_success:   Dict[int, list] = defaultdict(list)
        self._def_reward:    Dict[int, list] = defaultdict(list)
        self._atk_seq:       Dict[int, list] = defaultdict(list)  # attack sequences

        self._last_attack: Optional[int] = None

    # ──────────────────────────────────────────────────────────────────
    # Core update
    # ──────────────────────────────────────────────────────────────────
    def add_event(self,
                  attack_type:    int,
                  defense_action: int,
                  attack_success: float,
                  defense_reward: float,
                  step:           int) -> None:
        """Record one attack-defense interaction."""
        a_node = f"A_{attack_type}"
        d_node = f"D_{defense_action}"

        # Ensure nodes exist with metadata
        for node, ntype in [(a_node, 'attack'), (d_node, 'defense')]:
            if not self.G.has_node(node):
                self.G.add_node(node, ntype=ntype, count=0, score=0.0)
            self.G.nodes[node]['count'] += 1

        # Attack node: average success
        self._atk_success[attack_type].append(attack_success)
        self.G.nodes[a_node]['score'] = float(np.mean(self._atk_success[attack_type]))

        # Defense node: average reward
        self._def_reward[defense_action].append(defense_reward)
        self.G.nodes[d_node]['score'] = float(np.mean(self._def_reward[defense_action]))

        # Edge: attack → defense (interaction)
        self._update_edge(a_node, d_node,
                          weight_inc=1.0,
                          success=attack_success,
                          reward=defense_reward)

        # Edge: attack sequence  (prev attack → current attack)
        if self._last_attack is not None and self._last_attack != attack_type:
            prev_node = f"A_{self._last_attack}"
            self._update_edge(prev_node, a_node, weight_inc=1.0)
            self._atk_seq[self._last_attack].append(attack_type)

        # Edge: defense → next context  (defense → attack)
        if self._last_attack is not None:
            self._update_edge(d_node, a_node, weight_inc=0.5)

        self._last_attack = attack_type
        self.events.append({
            'step': step, 'attack': attack_type,
            'defense': defense_action,
            'a_success': attack_success, 'd_reward': defense_reward,
        })

    def _update_edge(self, src: str, dst: str,
                     weight_inc: float = 1.0,
                     **kwargs) -> None:
        if self.G.has_edge(src, dst):
            self.G[src][dst]['weight'] += weight_inc
            for k, v in kwargs.items():
                old = self.G[src][dst].get(k, v)
                self.G[src][dst][k] = old * 0.85 + v * 0.15   # EMA
        else:
            self.G.add_edge(src, dst, weight=weight_inc, **kwargs)

    # ──────────────────────────────────────────────────────────────────
    # Predictions
    # ──────────────────────────────────────────────────────────────────
    def predict_next_attack(self,
                            recent: List[int]) -> Tuple[int, float]:
        """
        Predict the most likely next attack type given recent attack history.
        Uses Markov-chain transition probabilities derived from the graph.
        Returns (attack_type, confidence ∈ [0, 1]).
        """
        if not recent:
            return 0, 0.0

        last_node = f"A_{recent[-1]}"
        if not self.G.has_node(last_node):
            return self._global_most_frequent()

        # Outgoing edges to other attack nodes
        transitions: Dict[int, float] = {}
        for nbr in self.G.successors(last_node):
            if nbr.startswith('A_'):
                transitions[int(nbr[2:])] = self.G[last_node][nbr]['weight']

        if not transitions:
            return self._global_most_frequent()

        total = sum(transitions.values())
        best  = max(transitions, key=transitions.get)
        conf  = transitions[best] / total
        return best, float(conf)

    def _global_most_frequent(self) -> Tuple[int, float]:
        """Fallback: return globally most observed attack type."""
        atk_counts = {
            int(n[2:]): self.G.nodes[n]['count']
            for n in self.G.nodes
            if n.startswith('A_')
        }
        if not atk_counts:
            return 0, 0.0
        best  = max(atk_counts, key=atk_counts.get)
        total = sum(atk_counts.values())
        return best, atk_counts[best] / total

    def get_best_defense(self, attack_type: int) -> int:
        """
        Return the historically most effective defense for a given attack type.
        Uses average EMA defense reward stored on A→D edges.
        """
        a_node = f"A_{attack_type}"
        if not self.G.has_node(a_node):
            return 0

        best_def, best_r = 0, -np.inf
        for nbr in self.G.successors(a_node):
            if nbr.startswith('D_'):
                r = self.G[a_node][nbr].get('reward', 0.0)
                if r > best_r:
                    best_r = r
                    best_def = int(nbr[2:])
        return best_def

    def get_proactive_defense(self,
                              recent: List[int]) -> Optional[int]:
        """
        If prediction confidence ≥ PROACTIVE_THRESHOLD,
        return the best preemptive defense action, else None.
        """
        next_atk, conf = self.predict_next_attack(recent)
        if conf >= PROACTIVE_THRESHOLD:
            return self.get_best_defense(next_atk)
        return None

    # ──────────────────────────────────────────────────────────────────
    # Analytics
    # ──────────────────────────────────────────────────────────────────
    def pagerank_scores(self) -> Dict[str, float]:
        """PageRank on attack nodes — higher = more 'central' attack."""
        if self.G.number_of_nodes() < 2:
            return {}
        try:
            pr = nx.pagerank(self.G, weight='weight')
            return {k: v for k, v in pr.items() if k.startswith('A_')}
        except Exception:
            return {}

    def summary(self) -> Dict:
        n_a = sum(1 for n in self.G.nodes if n.startswith('A_'))
        n_d = sum(1 for n in self.G.nodes if n.startswith('D_'))
        return {
            'attack_nodes':  n_a,
            'defense_nodes': n_d,
            'total_edges':   self.G.number_of_edges(),
            'events_logged': len(self.events),
        }

    def attack_transition_matrix(self) -> np.ndarray:
        """8×8 transition probability matrix between attack types."""
        M = np.zeros((8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                a, b = f"A_{i}", f"A_{j}"
                if self.G.has_edge(a, b):
                    M[i, j] = self.G[a][b]['weight']
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return M / row_sums

    def defense_effectiveness_matrix(self) -> np.ndarray:
        """8×8 matrix of avg defense rewards per (attack, defense) pair."""
        M = np.full((8, 8), np.nan, dtype=np.float32)
        for i in range(8):
            for j in range(8):
                a, d = f"A_{i}", f"D_{j}"
                if self.G.has_edge(a, d):
                    M[i, j] = self.G[a][d].get('reward', 0.0)
        return M
