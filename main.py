#!/usr/bin/env python3
"""
CAESAR Framework — Main Simulation Script
==========================================
Run: python main.py [--episodes 150] [--steps 50] [--seed 42]

Hardware recommendations
  • GTX 1650 / 16 GB RAM  →  episodes ≤ 200, steps ≤ 50  (fast, local)
  • Google Colab Pro       →  episodes up to 500, larger models
"""

import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from caesar import (
    CyberEnvironment, CAESAR,
    compute_episode_metrics, print_report, generate_all,
)
from caesar.metrics import per_attack_breakdown


# ═══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="CAESAR Framework")
    p.add_argument('--episodes',  type=int, default=150,
                   help='Number of training episodes (default: 150)')
    p.add_argument('--steps',     type=int, default=50,
                   help='Steps per episode (default: 50)')
    p.add_argument('--nodes',     type=int, default=10,
                   help='Number of simulated network nodes (default: 10)')
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--eval-eps',  type=int, default=20,
                   help='Evaluation episodes (default: 20)')
    p.add_argument('--out',       type=str, default='results/',
                   help='Output directory for results and figures')
    p.add_argument('--save',      action='store_true',
                   help='Save model checkpoints')
    p.add_argument('--quiet',     action='store_true',
                   help='Suppress per-episode logging')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "╔" + "═" * 63 + "╗")
    print("║  CAESAR: Co-Evolutionary Adversarial Simulation Engine      ║")
    print("║  Novel Algorithm — PhD Research Framework v1.0              ║")
    print("╚" + "═" * 63 + "╝\n")
    print(f"  Device   : {device}")
    print(f"  Episodes : {args.episodes}  |  Steps/ep : {args.steps}")
    print(f"  Nodes    : {args.nodes}  |  Seed : {args.seed}\n")

    # ── Environment ──────────────────────────────────────────────────
    env = CyberEnvironment(n_nodes=args.nodes, seed=args.seed)

    # ── CAESAR ───────────────────────────────────────────────────────
    caesar = CAESAR(env=env, device=device)

    # ── Training ─────────────────────────────────────────────────────
    logs = caesar.train(
        n_episodes=args.episodes,
        n_steps=args.steps,
        verbose=not args.quiet,
    )

    # ── Evaluation ───────────────────────────────────────────────────
    eval_results = caesar.evaluate(
        n_episodes=args.eval_eps,
        n_steps=args.steps,
    )

    # ── Metrics ──────────────────────────────────────────────────────
    train_metrics = compute_episode_metrics(logs)
    print_report(train_metrics, title="CAESAR Training Report")
    print_report(eval_results,  title="CAESAR Evaluation Report")

    # Per-attack breakdown
    breakdown = per_attack_breakdown(logs, env.attack_log)
    print("\n  Per-Attack-Type Breakdown:")
    for atype, stats in sorted(breakdown.items()):
        from caesar.environment import ATTACK_NAMES
        name = ATTACK_NAMES[atype] if atype < len(ATTACK_NAMES) else str(atype)
        print(f"    {name:<16s} n={stats['n_attacks']:4d}  "
              f"mean_success={stats['mean_success']:.3f}  "
              f"neutralized={stats['neutralization']:.3f}")

    # ── Save results ─────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/train_metrics.json", 'w') as f:
        json.dump(train_metrics, f, indent=2, default=str)
    with open(f"{args.out}/eval_metrics.json", 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    with open(f"{args.out}/episode_logs.json", 'w') as f:
        json.dump(logs, f, indent=2, default=str)

    if args.save:
        caesar.save('checkpoints/')

    # ── Visualisations ───────────────────────────────────────────────
    figure_paths = generate_all(
        logs=logs,
        tag=caesar.tag,
        eval_metrics=eval_results,
        save_dir=args.out,
    )

    print(f"\n  All results saved to: {args.out}/")
    print("  Figures:")
    for p in figure_paths:
        print(f"    • {p}")
    print("\n  CAESAR simulation complete. ✓\n")


if __name__ == '__main__':
    main()
