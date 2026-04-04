"""
CAESAR — Visualization Suite
==============================
Generates publication-quality figures for the PhD thesis.

Figures produced
────────────────
1. training_curves.png   — defense reward, attack success, health over episodes
2. coevo_fitness.png     — attacker vs defender fitness (co-evolutionary gap)
3. threat_heatmaps.png   — attack transition matrix + defense effectiveness matrix
4. threat_graph.png      — TAG network visualisation
5. metrics_summary.png   — bar chart of evaluation metrics
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .environment  import ATTACK_NAMES, DEFENSE_NAMES
from .threat_graph import TemporalAttackGraph

SAVE_DPI = 150
PALETTE  = {
    'attack':   '#e74c3c',
    'defense':  '#2980b9',
    'health':   '#27ae60',
    'proact':   '#f39c12',
    'neutral':  '#8e44ad',
    'bg':       '#0d1117',
    'panel':    '#161b22',
    'text':     '#c9d1d9',
    'grid':     '#21262d',
}


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PALETTE['panel'])
    ax.tick_params(colors=PALETTE['text'])
    ax.xaxis.label.set_color(PALETTE['text'])
    ax.yaxis.label.set_color(PALETTE['text'])
    ax.title.set_color(PALETTE['text'])
    for spine in ax.spines.values():
        spine.set_color(PALETTE['grid'])
    ax.grid(True, color=PALETTE['grid'], linewidth=0.6, alpha=0.8)
    if title:   ax.set_title(title, fontsize=11, fontweight='bold')
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9)


def _smooth(arr, w=10):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w) / w, mode='valid')


# ═══════════════════════════════════════════════════════════════════════
def plot_training_curves(logs: List[Dict], save_dir: str = 'results/') -> str:
    ep = np.arange(1, len(logs) + 1)
    dr  = [m['avg_defense_reward']  for m in logs]
    asr = [m['avg_attack_success']  for m in logs]
    hlt = [m['network_health']      for m in logs]
    pro = [m['n_proactive']         for m in logs]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                             facecolor=PALETTE['bg'])
    fig.suptitle('CAESAR Training Dynamics', color=PALETTE['text'],
                 fontsize=14, fontweight='bold', y=0.98)

    # Defense reward
    ax = axes[0, 0]
    ax.plot(ep, dr, color=PALETTE['defense'], alpha=0.25, lw=1)
    sw = min(15, len(dr))
    ax.plot(ep[sw-1:], _smooth(dr, sw), color=PALETTE['defense'], lw=2.2,
            label='Smoothed')
    _style_ax(ax, 'Average Defense Reward', 'Episode', 'Reward')
    ax.legend(fontsize=8, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])

    # Attack success
    ax = axes[0, 1]
    ax.fill_between(ep, asr, alpha=0.25, color=PALETTE['attack'])
    ax.plot(ep, asr, color=PALETTE['attack'], lw=1.5)
    ax.plot(ep[sw-1:], _smooth(asr, sw), color='white', lw=1.8, ls='--',
            label='Smoothed')
    _style_ax(ax, 'Attack Success Rate', 'Episode', 'Success ∈ [0,1]')
    ax.legend(fontsize=8, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])

    # Network health
    ax = axes[1, 0]
    ax.fill_between(ep, hlt, alpha=0.20, color=PALETTE['health'])
    ax.plot(ep, hlt, color=PALETTE['health'], lw=2)
    ax.axhline(0.5, color='white', ls=':', lw=1, alpha=0.5, label='threshold=0.5')
    _style_ax(ax, 'Network Health', 'Episode', 'Health ∈ [0,1]')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])

    # Proactive defenses
    ax = axes[1, 1]
    ax.bar(ep, pro, color=PALETTE['proact'], alpha=0.75, width=0.9)
    _style_ax(ax, 'Proactive Defenses / Episode', 'Episode', 'Count')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f'{save_dir}/training_curves.png'
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════
def plot_coevo_fitness(logs: List[Dict], save_dir: str = 'results/') -> str:
    ep  = np.arange(1, len(logs) + 1)
    f_a = [m['attacker_fitness']  for m in logs]
    f_d = [m['defender_fitness']  for m in logs]
    gap = np.array(f_d) - np.array(f_a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   facecolor=PALETTE['bg'])
    fig.suptitle('Co-evolutionary Fitness — CAESAR', color=PALETTE['text'],
                 fontsize=13, fontweight='bold')

    ax1.plot(ep, f_a, color=PALETTE['attack'],  lw=2, label='Attacker Fitness')
    ax1.plot(ep, f_d, color=PALETTE['defense'], lw=2, label='Defender Fitness')
    ax1.fill_between(ep, f_a, f_d,
                     where=np.array(f_d) > np.array(f_a),
                     alpha=0.2, color=PALETTE['defense'], label='Defender leading')
    ax1.fill_between(ep, f_a, f_d,
                     where=np.array(f_a) >= np.array(f_d),
                     alpha=0.2, color=PALETTE['attack'], label='Attacker leading')
    _style_ax(ax1, 'Fitness over Training', 'Episode', 'Fitness Score')
    ax1.legend(fontsize=8, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])

    ax2.fill_between(ep, gap, 0, where=(gap >= 0),
                     alpha=0.4, color=PALETTE['defense'], label='Defender advantage')
    ax2.fill_between(ep, gap, 0, where=(gap < 0),
                     alpha=0.4, color=PALETTE['attack'], label='Attacker advantage')
    ax2.plot(ep, gap, color='white', lw=1.2)
    ax2.axhline(0, color='gray', ls='--', lw=1)
    _style_ax(ax2, 'Co-evolutionary Gap (F_def − F_att)', 'Episode', 'Gap')
    ax2.legend(fontsize=8, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f'{save_dir}/coevo_fitness.png'
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════
def plot_threat_heatmaps(tag: TemporalAttackGraph,
                         save_dir: str = 'results/') -> str:
    T = tag.attack_transition_matrix()
    E = tag.defense_effectiveness_matrix()

    atk_labels = [a.replace('_', ' ')  for a in ATTACK_NAMES]
    def_labels  = [d.replace('_', '\n') for d in DEFENSE_NAMES]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   facecolor=PALETTE['bg'])
    fig.suptitle('Threat Intelligence Heatmaps', color=PALETTE['text'],
                 fontsize=13, fontweight='bold')

    # Transition matrix
    im1 = ax1.imshow(T, cmap='Reds', vmin=0, vmax=1)
    ax1.set_xticks(range(8)); ax1.set_yticks(range(8))
    ax1.set_xticklabels(atk_labels, rotation=35, ha='right',
                        fontsize=8, color=PALETTE['text'])
    ax1.set_yticklabels(atk_labels, fontsize=8, color=PALETTE['text'])
    ax1.set_title('Attack Transition Probabilities', color=PALETTE['text'],
                  fontsize=11, fontweight='bold')
    ax1.set_facecolor(PALETTE['panel'])
    plt.colorbar(im1, ax=ax1)

    # Defense effectiveness
    E_plot = np.where(np.isnan(E), 0, E)
    im2 = ax2.imshow(E_plot, cmap='Blues', vmin=0, vmax=1)
    ax2.set_xticks(range(8)); ax2.set_yticks(range(8))
    ax2.set_xticklabels(def_labels, rotation=35, ha='right',
                        fontsize=7, color=PALETTE['text'])
    ax2.set_yticklabels(atk_labels, fontsize=8, color=PALETTE['text'])
    ax2.set_title('Defense Effectiveness per Attack Type',
                  color=PALETTE['text'], fontsize=11, fontweight='bold')
    ax2.set_facecolor(PALETTE['panel'])
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f'{save_dir}/threat_heatmaps.png'
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════
def plot_metrics_bar(eval_metrics: Dict, save_dir: str = 'results/') -> str:
    keys_to_plot = [
        'mean_defense_reward', 'mean_attack_success',
        'mean_detection_rate', 'robustness_score',
        'neutralization_rate', 'co_evolutionary_gap',
        'mean_network_health',
    ]
    labels = [k.replace('_', '\n') for k in keys_to_plot]
    values = [float(eval_metrics.get(k, 0.0)) for k in keys_to_plot]
    colors = [PALETTE['defense'], PALETTE['attack'], PALETTE['health'],
              PALETTE['neutral'], PALETTE['proact'], PALETTE['defense'],
              PALETTE['health']]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE['bg'])
    bars = ax.bar(labels, values, color=colors, alpha=0.80, edgecolor='white',
                  linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=8, color=PALETTE['text'])
    _style_ax(ax, 'CAESAR Evaluation Metrics Summary', '', 'Score')
    ax.set_facecolor(PALETTE['panel'])
    ax.tick_params(axis='x', colors=PALETTE['text'], labelsize=8)

    plt.tight_layout()
    out = f'{save_dir}/metrics_summary.png'
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════
def generate_all(logs:         List[Dict],
                 tag:          TemporalAttackGraph,
                 eval_metrics: Dict,
                 save_dir:     str = 'results/') -> List[str]:
    """Generate and save all figures. Returns list of file paths."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    print("  Generating visualisations...")
    paths.append(plot_training_curves(logs, save_dir))
    paths.append(plot_coevo_fitness(logs, save_dir))
    paths.append(plot_threat_heatmaps(tag, save_dir))
    paths.append(plot_metrics_bar(eval_metrics, save_dir))
    for p in paths:
        print(f"    ✓ {p}")
    return paths
