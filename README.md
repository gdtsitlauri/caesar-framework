# CAESAR Framework — Complete PhD Research Package
## Co-Evolutionary Adversarial Simulation Engine for Attack & Response

**Author:** George David Tsitlauri  
**Affiliation:** Dept. of Informatics & Telecommunications, University of Thessaly, Greece  
**Contact:** gdtsitlauri@gmail.com  
**Year:** 2026

---

## 📁 Complete File Structure

```
caesar-framework/
│
├── 📂 caesar/                          # Core Python package
│   ├── __init__.py
│   ├── environment.py                  # CyberEnvironment simulation
│   ├── ta_gan.py                       # TA-GAN (PyTorch)
│   ├── adpn.py                         # ADPN Dueling Double-DQN (PyTorch)
│   ├── threat_graph.py                 # Temporal Attack Graph
│   ├── caesar_algorithm.py             # Core co-evolutionary loop (PyTorch)
│   ├── metrics.py                      # Robustness, neutralization, coevo metrics
│   ├── visualization.py               # Publication figures
│   ├── dataset.py                      # CICIDS2017 loader + synthetic generator
│   ├── baselines.py                    # RF IDS, DT IDS, Threshold IDS
│   ├── explainability.py              # Permutation importance, Q-attribution
│   ├── phishing_module.py             # LLM phishing generator + detector
│   ├── diffusion_module.py            # MGAE Diffusion engine (Novel #5)
│   └── self_healing.py                # ARL Self-healing system (Novel #6)
│
├── 📂 thesis/                          # LaTeX PhD Thesis
│   ├── main.tex                        # Root document
│   ├── references.bib                  # 18 references
│   └── chapters/
│       ├── 00_abstract.tex
│       ├── 01_introduction.tex         # ✅ Full content
│       ├── 02_background.tex           # ✅ Full content (GANs, DQN, DDPM, graphs)
│       ├── 03_related_work.tex
│       ├── 04_framework.tex            # ✅ Full content + TikZ architecture + Algorithm
│       ├── 05_tagan.tex
│       ├── 06_adpn.tex
│       ├── 07_tag.tex
│       ├── 08_mgae.tex
│       ├── 09_self_healing.tex
│       ├── 10_experiments.tex          # ✅ Full content + all tables
│       ├── 11_discussion.tex
│       ├── 12_conclusion.tex           # ✅ Full content
│       ├── A_code.tex
│       └── B_datasets.tex
│
├── 📂 paper/
│   └── caesar_paper.tex               # ✅ Full IEEE conference paper
│
├── 📂 colab/
│   └── CAESAR_Complete.ipynb          # ✅ Full Colab notebook (GPU-ready)
│
├── 📂 results/                         # Generated figures (15 total)
│   ├── training_curves.png
│   ├── coevo_fitness.png
│   ├── threat_heatmaps.png
│   ├── metrics_summary.png
│   ├── per_attack_breakdown.png
│   ├── comparison_models.png
│   ├── xai_feature_importance.png
│   ├── xai_q_attribution.png
│   ├── xai_decision_rules.png
│   ├── phishing_analysis.png
│   ├── robustness_evaluation.png
│   ├── mgae_diffusion.png
│   ├── self_healing_timeline.png
│   ├── mgae_sensitivity.png
│   └── self_healing_summary.png
│
├── caesar_demo.py                      # ✅ NumPy standalone (no GPU needed)
├── main.py                             # ✅ PyTorch entry point
├── phase2_run.py                       # ✅ Phase 2 pipeline
├── phase3_run.py                       # ✅ Phase 3 pipeline
├── dashboard.html                      # ✅ Live interactive dashboard
└── requirements.txt
```

---

## 🆕 Novel Contributions (6 total)

| # | Name | Type | Location |
|---|------|------|----------|
| 1 | **TA-GAN** | Defense-conditioned attack generator | `caesar/ta_gan.py` |
| 2 | **ADPN** | Dueling Double-DQN adaptive defender | `caesar/adpn.py` |
| 3 | **TAG** | Temporal Attack Graph + proactive defense | `caesar/threat_graph.py` |
| 4 | **CAESAR Loop** | Co-evolutionary fitness function | `caesar/caesar_algorithm.py` |
| 5 | **MGAE** | Manifold-Guided Adversarial Engine (Diffusion) | `caesar/diffusion_module.py` |
| 6 | **ARL** | Autonomous Remediation Loop (Self-healing) | `caesar/self_healing.py` |

---

## 🚀 Running Everything

### Local (NumPy, no GPU):
```bash
python3 caesar_demo.py          # Phase 1: CAESAR core
python3 phase2_run.py           # Phase 2: Baselines + XAI + Phishing
python3 phase3_run.py           # Phase 3: Diffusion + Robustness + Self-healing
open dashboard.html             # Live interactive dashboard
```

### Local (PyTorch GPU):
```bash
pip install -r requirements.txt
python main.py --episodes 500 --steps 100 --save
```

### Google Colab (GPU T4/A100):
1. Upload `CAESAR_Complete.ipynb`
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells
4. Download `caesar_results.zip`

### With real CICIDS2017 data:
```bash
# Kaggle download:
pip install kaggle
kaggle datasets download -d cicdataset/cicids2017 -p data/
python main.py --data data/cicids2017.csv --episodes 500
```

---

## 📊 Full Results Summary

| Phase | Metric | Value |
|-------|--------|-------|
| 1 | Neutralization Rate | **96.5%** |
| 1 | Robustness Score | **0.951** |
| 1 | Co-evolutionary Gap | **+0.262** |
| 1 | TAG edges learned | 120 |
| 1 | Proactive defenses | 27 |
| 2 | RF IDS F1 | 1.000 |
| 2 | DT IDS F1 | 1.000 |
| 2 | Threshold IDS F1 | 0.712 |
| 2 | CAESAR F1 (adaptive) | **0.977** |
| 2 | Phishing detection DR | **0.857** |
| 2 | Phishing evasion (perturbed) | 0.220 |
| 3 | MGAE evasion prob | **41.4%** |
| 3 | Self-healing success | **100%** |
| 3 | Mean network health (ARL) | **76.4%** |
| 3 | Human escalations | **0** |

---

## 📚 Target Publications

| Venue | Tier | Deadline |
|-------|------|----------|
| IEEE S&P (Oakland) | A* | Nov |
| USENIX Security | A* | Oct |
| NDSS | A* | Jul |
| CCS | A* | Jan |
| IEEE TIFS (journal) | Q1 | Rolling |
| Computers & Security | Q1 | Rolling |

---

## 📝 PhD Timeline

| Year | Phase | Milestone |
|------|-------|-----------|
| Y1-Q1 | Literature Review | Background, RQ formulation |
| Y1-Q2 | Phase 1 | CAESAR core, TA-GAN, ADPN |
| Y1-Q3 | Phase 1 | TAG, experiments, first paper |
| Y1-Q4 | Phase 2 | Baselines, XAI, phishing module |
| Y2-Q1 | Phase 3 | MGAE, robustness evaluation |
| Y2-Q2 | Phase 3 | Self-healing ARL, second paper |
| Y2-Q3 | Writing | Thesis Chapters 1-6 |
| Y2-Q4 | Writing | Thesis Chapters 7-12 |
| Y3-Q1 | Revision | Thesis revisions, viva prep |
| Y3-Q2 | Submission | Final submission + viva |

---

*CAESAR Framework — PhD Research Package · Version 1.0*

## Citation

```bibtex
@misc{tsitlauri2026caesar,
  author = {George David Tsitlauri},
  title  = {CAESAR: Co-Evolutionary Adversarial Simulation Engine for Attack \& Response},
  year   = {2026},
  institution = {University of Thessaly},
  email  = {gdtsitlauri@gmail.com}
}
```
