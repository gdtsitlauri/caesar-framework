# CAESAR

**Co-Evolutionary Adversarial Simulation Engine for Attack & Response**

An AI-driven cybersecurity framework that simultaneously simulates realistic cyber-attacks and trains adaptive defenses through a closed co-evolutionary loop. CAESAR detects threats, responds autonomously, and self-heals — without human intervention.

---

## What It Does

CAESAR puts an AI attacker against an AI defender inside a simulated network. Both evolve together: the attacker learns to bypass defenses, the defender learns to stop new attacks. The result is a defense system that handles threats it has never seen before.

### 6 Novel Algorithms

| Algorithm | What it does |
|-----------|-------------|
| **TA-GAN** | Generates attacks that adapt to deployed defenses in real-time |
| **ADPN** | Dueling Double-DQN agent that selects the best countermeasure |
| **TAG** | Temporal graph that predicts the next attack before it arrives |
| **CAESAR Loop** | Co-evolutionary engine linking attacker and defender fitness |
| **MGAE** | Diffusion model that makes attacks invisible to classifiers |
| **ARL** | Self-healing state machine: detect, respond, heal, verify |

---

## Results (Real CICIDS2017 Data — 2.5M flows)

| Metric | Value |
|--------|-------|
| Neutralization Rate | **91.9% +/- 3.3%** |
| Robustness Score | **0.967 +/- 0.007** |
| Co-evolutionary Gap | **+0.239** (defender wins) |
| MGAE vs Random Forest | 98.5% -> **0% detection** |
| MGAE vs Decision Tree | 99.5% -> **0% detection** |
| Self-Healing Success | **100%** |
| Human Interventions | **0** |
| Statistical Significance | Wilcoxon p=0.002, Cohen's d=23.3 |

---

## Quick Start

### Install
```bash
git clone https://github.com/gdtsitlauri/caesar-framework.git
cd caesar-framework
pip install -r requirements.txt
```

### Run (no GPU needed)
```bash
python caesar_demo.py          # Core co-evolutionary training
python phase2_run.py           # Baselines + XAI + Phishing detection
python phase3_run.py           # Adversarial robustness + Self-healing
python statistical_eval.py     # 10-seed statistical validation
```

### Live Dashboard
Open `dashboard.html` in your browser — real-time simulation visualization.

### With Real CICIDS2017 Data
Download from [Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017), place the CSV in the project root, and re-run the scripts. CAESAR auto-detects real data.

### Google Colab (GPU)
Upload `colab/CAESAR_Complete.ipynb` to Colab, select T4/A100 runtime, run all cells.

---

## Project Structure

```
caesar-framework/
|-- caesar/                    # Core Python package (14 modules)
|   |-- environment.py         # Network simulation environment
|   |-- ta_gan.py              # Threat-Aware GAN (PyTorch)
|   |-- adpn.py                # Adaptive Defense Policy Network
|   |-- threat_graph.py        # Temporal Attack Graph
|   |-- caesar_algorithm.py    # Co-evolutionary loop
|   |-- diffusion_module.py    # MGAE diffusion engine
|   |-- self_healing.py        # Autonomous Remediation Loop
|   |-- dataset.py             # CICIDS2017 loader
|   |-- baselines.py           # RF, DT, Threshold, IDSGAN, WGAN, DRL baselines
|   |-- explainability.py      # Feature importance + Q-attribution
|   |-- phishing_module.py     # Phishing generator + detector
|   |-- metrics.py             # Evaluation metrics
|   |-- visualization.py       # Publication-quality figures
|   +-- __init__.py
|
|-- results/                   # Generated figures + reports
|-- colab/                     # Colab notebook
|-- caesar_demo.py             # Standalone demo (NumPy only)
|-- main.py                    # Full PyTorch entry point
|-- phase2_run.py              # Phase 2 pipeline
|-- phase3_run.py              # Phase 3 pipeline
|-- statistical_eval.py        # Statistical validation
|-- dashboard.html             # Interactive browser dashboard
+-- requirements.txt
```

---

## Baselines Compared

| Model | F1 | Detection Rate | Robustness |
|-------|-----|---------------|------------|
| Random Forest IDS | 0.995 | 0.995 | Fails under MGAE |
| Decision Tree IDS | 0.993 | 0.998 | Fails under MGAE |
| Threshold IDS | 0.383 | 0.754 | Partial |
| IDSGAN (Lin et al.) | 0.996 | 0.995 | Fails under MGAE |
| WGAN-IDS (Ring et al.) | 0.996 | 0.996 | Fails under MGAE |
| Deep RL Defender | 0.715 | 0.599 | Partial |
| **CAESAR** | **0.977** | **0.421** | **0.967** |

CAESAR's strength is **adaptive robustness** — it keeps working when adversarial attacks break all other models.

---

## Requirements

- Python 3.10+
- NumPy, Matplotlib, scikit-learn, pandas, scipy, networkx
- PyTorch 2.0+ (optional, for GPU training)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

This framework is for **defensive cybersecurity research only**.

---

## Author

**George David Tsitlauri** — University of Thessaly, Department of Computer Science

Contact: gdtsitlauri@gmail.com
