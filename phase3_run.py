#!/usr/bin/env python3
"""CAESAR Phase 3 — Diffusion + Robustness + Self-Healing"""

import os, sys, json, importlib.util
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

sys.path.insert(0, '.')
sys.path.insert(0, './caesar')

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m    = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

_env  = _load('environment',      'caesar/environment.py')
_ds   = _load('dataset',          'caesar/dataset.py')
_bl   = _load('baselines',        'caesar/baselines.py')
_diff = _load('diffusion_module', 'caesar/diffusion_module.py')
_sh   = _load('self_healing',     'caesar/self_healing.py')

from caesar_demo import CyberEnv, CAESAR, compute_metrics, PAL, _sa, _sm
CICIDSDataset      = _ds.CICIDSDataset
RandomForestIDS    = _bl.RandomForestIDS
DecisionTreeIDS    = _bl.DecisionTreeIDS
ThresholdIDS       = _bl.ThresholdIDS
MGAEEngine         = _diff.MGAEEngine
RobustnessEvaluator= _diff.RobustnessEvaluator
SelfHealingSystem  = _sh.SelfHealingSystem

def _q_values(self, state):
    v = self.value_net.forward(state)
    a = self.adv_net.forward(state)
    return v + a - a.mean()

from caesar_demo import ADPN_NP
ADPN_NP.q_values = _q_values

OUT = 'results/'
os.makedirs(OUT, exist_ok=True)

print("\n╔" + "═"*63 + "╗")
print("║  CAESAR Phase 3 — Diffusion · Robustness · Self-Healing       ║")
print("╚" + "═"*63 + "╝\n")

# ══════════════════════════════════════════════════════════════════════
# Step 1 — Dataset + Baselines
# ══════════════════════════════════════════════════════════════════════
print("  Step 1: Dataset + baseline IDS...")
ds = CICIDSDataset(n_synthetic=20_000, seed=42).load()
# Subsample large datasets for tractable training
MAX_SAMPLES = 200_000
if len(ds.df) > MAX_SAMPLES:
    print(f"    Subsampling {len(ds.df):,} → {MAX_SAMPLES:,} (stratified)...")
    n_total = len(ds.df)
    ds.df = ds.df.sample(n=MAX_SAMPLES, random_state=42, replace=False).reset_index(drop=True)
X_tr, X_te, y_tr, y_te = ds.split()
X_tr_n = ds.fit_transform(X_tr)
X_te_n = ds.transform(X_te)

rf  = RandomForestIDS(n_estimators=60, seed=42).fit(X_tr_n, y_tr)
dt  = DecisionTreeIDS(max_depth=12, seed=42).fit(X_tr_n, y_tr)
thr = ThresholdIDS(z_threshold=2.0).fit(X_tr_n, y_tr)
print("    Baselines trained.")

# ══════════════════════════════════════════════════════════════════════
# Step 2 — MGAE Diffusion Engine
# ══════════════════════════════════════════════════════════════════════
print("\n  Step 2: Training MGAE diffusion engine...")
X_benign = X_te_n[y_te == 0][:800]
X_attack = X_te_n[y_te == 1][:200]

mgae = MGAEEngine(feat_dim=X_benign.shape[1], T=40, seed=42)
mgae.fit(X_benign, n_epochs=40)

# ══════════════════════════════════════════════════════════════════════
# Step 3 — Robustness Evaluation
# ══════════════════════════════════════════════════════════════════════
print("\n  Step 3: Adversarial robustness evaluation...")
rob_eval = RobustnessEvaluator()
results  = []
for model in [rf, dt, thr]:
    r = rob_eval.evaluate(model, X_attack, mgae, epsilon=0.15)
    results.append(r)
    print(f"    {r['model']:<35} Clean:{r['dr_clean']:.3f} "
          f"FGSM:{r['dr_fgsm']:.3f} PGD:{r['dr_pgd']:.3f} MGAE:{r['dr_mgae']:.3f}")

print(f"\n    MGAE metrics: L2={results[0]['mgae_l2']:.3f}  "
      f"Linf={results[0]['mgae_linf']:.3f}  "
      f"Manifold={results[0]['mgae_manifold']:.3f}  "
      f"Evasion={results[0]['mgae_evasion']:.3f}")

# ══════════════════════════════════════════════════════════════════════
# Step 4 — CAESAR Training + Self-Healing
# ══════════════════════════════════════════════════════════════════════
print("\n  Step 4: CAESAR + self-healing simulation...")
env    = CyberEnv(n_nodes=10, seed=42)
caesar = CAESAR(env)
caesar.train(150, 50, verbose=False)

shs = SelfHealingSystem(caesar, env)
heal_log = shs.run_simulation(n_ticks=300, verbose=True)
heal_summary = shs.summary()
print(f"\n    Healing Summary: {heal_summary}")

# ══════════════════════════════════════════════════════════════════════
# Step 5 — All Phase 3 Figures
# ══════════════════════════════════════════════════════════════════════
print("\n  Step 5: Generating Phase 3 figures...")

# ── Fig 1: Robustness comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PAL['bg'])
fig.suptitle('CAESAR Phase 3 — Adversarial Robustness Evaluation',
             color=PAL['txt'], fontsize=13, fontweight='bold')

attack_types = ['Clean', 'FGSM', 'PGD', 'MGAE']
colors_a     = [PAL['hlth'], PAL['pro'], PAL['df'], PAL['atk']]
x = np.arange(len(results))
bw = 0.18

ax = axes[0]
for ki, (at, col) in enumerate(zip(attack_types, colors_a)):
    key  = f'dr_{at.lower()}'
    vals = [r.get(key, 0) for r in results]
    off  = (ki - 1.5) * bw
    bars = ax.bar(x + off, vals, width=bw, color=col, alpha=.83,
                  edgecolor='white', lw=.4, label=at)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{val:.2f}', ha='center', fontsize=6.5, color=PAL['txt'])
ax.set_xticks(x)
ax.set_xticklabels([r['model'].replace(' IDS','') for r in results],
                   fontsize=9, color=PAL['txt'])
ax.set_ylim(0, 1.15)
_sa(ax, 'Detection Rate Under Adversarial Attack', '', 'Detection Rate')
ax.legend(fontsize=8, facecolor=PAL['panel'], labelcolor=PAL['txt'])

# Robustness drop bars
ax2 = axes[1]
drop_keys = ['drop_fgsm','drop_pgd','drop_mgae']
drop_cols  = [PAL['pro'], PAL['df'], PAL['atk']]
drop_names = ['FGSM Drop', 'PGD Drop', 'MGAE Drop']
for ki, (dk, col, dn) in enumerate(zip(drop_keys, drop_cols, drop_names)):
    vals = [r.get(dk, 0) for r in results]
    off  = (ki - 1) * 0.22
    bars = ax2.bar(x + off, vals, width=0.20, color=col, alpha=.83,
                   edgecolor='white', lw=.4, label=dn)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.2f}', ha='center', fontsize=6.5, color=PAL['txt'])
ax2.set_xticks(x)
ax2.set_xticklabels([r['model'].replace(' IDS','') for r in results],
                    fontsize=9, color=PAL['txt'])
_sa(ax2, 'Robustness Drop (DR_clean − DR_perturbed)', '', 'Drop')
ax2.legend(fontsize=8, facecolor=PAL['panel'], labelcolor=PAL['txt'])
ax2.axhline(0, color='white', ls='--', lw=0.8, alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
rob_fig = f'{OUT}robustness_evaluation.png'
plt.savefig(rob_fig, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
plt.close(fig)
print(f"    ✓ {rob_fig}")

# ── Fig 2: Diffusion training loss + perturbation analysis ───────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=PAL['bg'])
fig.suptitle('CAESAR Phase 3 — MGAE Diffusion Engine',
             color=PAL['txt'], fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(mgae.losses, color=PAL['df'], lw=2)
w = min(8, len(mgae.losses))
if len(mgae.losses) >= w:
    sm = np.convolve(mgae.losses, np.ones(w)/w, mode='valid')
    ax.plot(range(w-1, len(mgae.losses)), sm, color='white', lw=1.5, ls='--')
_sa(ax, 'Diffusion Score Network Training Loss', 'Epoch', 'MSE Loss')

# Perturbation magnitudes
ax = axes[1]
results_mgae_full = mgae.perturb_batch(X_attack[:100])
l2s   = [r.l2_distance   for r in results_mgae_full]
linfs = [r.linf_distance  for r in results_mgae_full]
mfld  = [r.in_manifold    for r in results_mgae_full]
evs   = [r.evasion_prob   for r in results_mgae_full]
ax.hist(l2s,   bins=20, color=PAL['df'],  alpha=.65, label='L2 dist')
ax.hist(linfs, bins=20, color=PAL['atk'], alpha=.65, label='L∞ dist')
_sa(ax, 'MGAE Perturbation Magnitudes', 'Distance', 'Count')
ax.legend(fontsize=8, facecolor=PAL['panel'], labelcolor=PAL['txt'])

# Evasion probability vs manifold alignment
ax = axes[2]
sc = ax.scatter(mfld, evs, c=l2s, cmap='plasma', alpha=0.6, s=25,
                vmin=min(l2s), vmax=max(l2s))
plt.colorbar(sc, ax=ax, label='L2 distance')
_sa(ax, 'Evasion Prob vs Manifold Alignment', 'Manifold Score', 'Evasion Prob')
ax.set_facecolor(PAL['panel'])

plt.tight_layout(rect=[0,0,1,0.95])
diff_fig = f'{OUT}mgae_diffusion.png'
plt.savefig(diff_fig, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
plt.close(fig)
print(f"    ✓ {diff_fig}")

# ── Fig 3: Self-healing state machine timeline ───────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=PAL['bg'])
fig.suptitle('CAESAR Phase 3 — Self-Healing System (ARL)',
             color=PAL['txt'], fontsize=13, fontweight='bold')

ticks = range(len(heal_log))
states  = [e['system_state'] for e in heal_log]
health  = shs.health_history
sevs    = [e['severity'] for e in heal_log]
state_map = {'NORMAL':0,'ALERT':1,'RESPONDING':2,'HEALING':3,'VERIFIED':4}
state_ids = [state_map.get(s,0) for s in states]

state_colors = {0:PAL['hlth'], 1:PAL['pro'], 2:PAL['atk'], 3:PAL['df'], 4:'#8e44ad'}
ax = axes[0]
for i, si in enumerate(state_ids):
    ax.bar(i, 1, color=state_colors[si], alpha=0.8, width=1.0)
ax.set_yticks([0.5])
ax.set_yticklabels(['State'], color=PAL['txt'])
_sa(ax, 'System State Timeline', '', '')
legend_patches = [mpatches.Patch(color=c, label=n)
                  for n, c in [('NORMAL',PAL['hlth']),('ALERT',PAL['pro']),
                                ('RESPONDING',PAL['atk']),('HEALING',PAL['df']),
                                ('VERIFIED','#8e44ad')]]
ax.legend(handles=legend_patches, fontsize=7, facecolor=PAL['panel'],
          labelcolor=PAL['txt'], loc='upper right', ncol=5)

ax = axes[1]
if health:
    ax.fill_between(range(len(health)), health, alpha=0.25, color=PAL['hlth'])
    ax.plot(health, color=PAL['hlth'], lw=2)
    ax.axhline(0.5, color='white', ls=':', lw=1, alpha=0.5)
    ax.set_ylim(0, 1.05)
    # Mark healing events
    for ev in shs.events:
        if ev.step < len(health):
            ax.axvline(ev.step, color=PAL['pro'], ls='--', lw=1, alpha=0.7)
_sa(ax, 'Network Health (dashed = healing event)', 'Tick', 'Health')

ax = axes[2]
ax.fill_between(ticks, sevs, alpha=0.3, color=PAL['atk'])
ax.plot(ticks, sevs, color=PAL['atk'], lw=1.2)
ax.axhline(SelfHealingSystem.RESPOND_THRESHOLD, color='white', ls=':', lw=1, alpha=0.6,
           label=f'Auto-respond threshold')
ax.axhline(SelfHealingSystem.ESCALATE_SEVERITY, color=PAL['pro'], ls='--', lw=1.2,
           label='Escalation threshold')
ax.set_ylim(0, 1.05)
_sa(ax, 'Attack Severity Over Time', 'Tick', 'Severity')
ax.legend(fontsize=8, facecolor=PAL['panel'], labelcolor=PAL['txt'])

plt.tight_layout(rect=[0,0,1,0.96])
heal_fig = f'{OUT}self_healing_timeline.png'
plt.savefig(heal_fig, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
plt.close(fig)
print(f"    ✓ {heal_fig}")

# ── Fig 4: MGAE t_inject sensitivity ────────────────────────────────
print("    Computing diffusion sensitivity curve...")
t_values = [5, 10, 15, 20, 25, 30]
dr_curve, l2_curve, ev_curve = [], [], []
X_probe = X_attack[:50]
for t in t_values:
    res_t  = mgae.perturb_batch(X_probe, t_inject=t)
    X_pert = np.array([r.perturbed for r in res_t])
    dr_t   = float(rf.predict(X_pert).mean())
    l2_t   = float(np.mean([r.l2_distance  for r in res_t]))
    ev_t   = float(np.mean([r.evasion_prob for r in res_t]))
    dr_curve.append(dr_t)
    l2_curve.append(l2_t)
    ev_curve.append(ev_t)

fig, ax1 = plt.subplots(figsize=(10, 5), facecolor=PAL['bg'])
ax2 = ax1.twinx()
l1, = ax1.plot(t_values, dr_curve, 'o-', color=PAL['atk'], lw=2.5, label='Detection Rate (DR)')
l2, = ax1.plot(t_values, ev_curve, 's--', color=PAL['pro'], lw=2, label='Evasion Probability')
l3, = ax2.plot(t_values, l2_curve, '^:', color=PAL['df'], lw=2, label='L2 Distance (right)')
ax1.set_xlabel('Diffusion Injection Step (t)', color=PAL['txt'])
ax1.set_ylabel('Score',      color=PAL['txt'])
ax2.set_ylabel('L2 Distance', color=PAL['df'])
ax1.tick_params(colors=PAL['txt'])
ax2.tick_params(colors=PAL['df'])
ax1.set_facecolor(PAL['panel'])
ax1.grid(True, color=PAL['grid'], lw=0.6, alpha=0.7)
for sp in ax1.spines.values(): sp.set_color(PAL['grid'])
ax1.set_title('MGAE Sensitivity: Injection Step vs. Evasion–Distortion Trade-off',
              color=PAL['txt'], fontsize=11, fontweight='bold')
ax1.legend(handles=[l1,l2,l3], fontsize=9, facecolor=PAL['panel'], labelcolor=PAL['txt'])
ax1.set_ylim(0, 1.05)
plt.tight_layout()
sens_fig = f'{OUT}mgae_sensitivity.png'
plt.savefig(sens_fig, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
plt.close(fig)
print(f"    ✓ {sens_fig}")

# ── Fig 5: Self-healing state distribution pie ───────────────────────
fig, (pa, pb) = plt.subplots(1, 2, figsize=(12, 5), facecolor=PAL['bg'])
fig.suptitle('CAESAR Phase 3 — Self-Healing System Summary',
             color=PAL['txt'], fontsize=12, fontweight='bold')

s  = heal_summary
pct_n  = s['pct_normal']
pct_r  = s['pct_responding']
pct_h  = s['pct_healing']
pct_o  = max(0., 1. - pct_n - pct_r - pct_h)
sizes  = [pct_n, pct_r, pct_h, pct_o]
lbls   = ['NORMAL','RESPONDING','HEALING','ALERT/VERIFIED']
clrs   = [PAL['hlth'], PAL['atk'], PAL['df'], PAL['pro']]
pa.pie(sizes, labels=lbls, colors=clrs, autopct='%1.1f%%', startangle=90,
       textprops={'color': PAL['txt'], 'fontsize': 9})
pa.set_title('Time in Each State', color=PAL['txt'], fontsize=11, fontweight='bold')
pa.set_facecolor(PAL['panel'])

keys_b = ['mean_health','pct_normal','successful_heals','avg_time_to_heal_ms']
lbs_b  = ['Mean Health','% Normal','Heals OK','Avg Heal ms']
vals_b = [s.get(k, 0) for k in keys_b]
# Normalise for display
vals_norm = [min(1., v) for v in vals_b]
bars_b = pb.bar(lbs_b, vals_norm,
                color=[PAL['hlth'],PAL['df'],PAL['pro'],PAL['atk']],
                alpha=0.82, edgecolor='white', lw=0.5)
for bar, raw in zip(bars_b, vals_b):
    pb.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f'{raw:.2f}' if raw < 10 else f'{raw:.0f}',
            ha='center', fontsize=9, color=PAL['txt'])
_sa(pb, 'Self-Healing KPIs', '', 'Score')
pb.set_ylim(0, 1.2)

plt.tight_layout(rect=[0,0,1,0.95])
pie_fig = f'{OUT}self_healing_summary.png'
plt.savefig(pie_fig, dpi=150, bbox_inches='tight', facecolor=PAL['bg'])
plt.close(fig)
print(f"    ✓ {pie_fig}")

# ══════════════════════════════════════════════════════════════════════
# Step 6 — Save report
# ══════════════════════════════════════════════════════════════════════
report = {
    'phase': 3,
    'mgae': {
        'feat_dim':   X_benign.shape[1],
        'T':          40,
        'final_loss': mgae.losses[-1],
        'mean_evasion_prob': float(np.mean(evs)),
        'mean_manifold':     float(np.mean(mfld)),
        'mean_l2':           float(np.mean(l2s)),
    },
    'robustness': {r['model']: {
        'dr_clean': r['dr_clean'], 'dr_fgsm': r['dr_fgsm'],
        'dr_pgd':   r['dr_pgd'],   'dr_mgae': r['dr_mgae'],
    } for r in results},
    'self_healing': heal_summary,
    'figures': [rob_fig, diff_fig, heal_fig, sens_fig, pie_fig],
}
with open(f'{OUT}phase3_report.json','w') as f:
    json.dump(report, f, indent=2)

print("\n" + "═"*65)
print("  CAESAR Phase 3 Summary")
print("═"*65)
print(f"  MGAE Evasion Rate   : {np.mean(evs):.3f}")
print(f"  Manifold Alignment  : {np.mean(mfld):.3f}")
print(f"  RF DR (clean→MGAE) : {results[0]['dr_clean']:.3f} → {results[0]['dr_mgae']:.3f}")
print(f"  Self-Healing Events : {heal_summary['healing_events']}")
print(f"  System Normal %     : {heal_summary['pct_normal']*100:.1f}%")
print(f"  Mean Network Health : {heal_summary['mean_health']:.3f}")
print("═"*65)
print("\n  Phase 3 complete. ✓\n")
