"""
CAESAR Phase 2 — Baseline IDS Models
======================================
Implements three classical IDS baselines for fair comparison with CAESAR:

  1. RandomForestIDS   — standard ensemble classifier
  2. DecisionTreeIDS   — interpretable single-tree classifier
  3. ThresholdIDS      — anomaly score threshold (no ML, pure rule-based)

All three share a common interface:
    .fit(X_train, y_train)
    .predict(X_test) → np.ndarray
    .evaluate(X_test, y_test) → Dict

This allows direct metric comparison in the evaluation report.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


# ─── shared evaluation helper ──────────────────────────────────────────
def _metrics(y_true: np.ndarray,
             y_pred:  np.ndarray,
             y_prob:  np.ndarray,
             label:   str) -> Dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0])
    fpr = fp / (fp + tn + 1e-9)
    dr  = recall_score(y_true, y_pred, zero_division=0)    # detection rate = recall

    return {
        'model':          label,
        'accuracy':       float(accuracy_score(y_true, y_pred)),
        'precision':      float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':         float(recall_score(y_true, y_pred, zero_division=0)),
        'f1':             float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc':        float(roc_auc_score(y_true, y_prob)),
        'detection_rate': float(dr),
        'false_pos_rate': float(fpr),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════
class RandomForestIDS:
    NAME = 'Random Forest IDS'

    def __init__(self, n_estimators: int = 100, max_depth: int = 20,
                 seed: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight='balanced', random_state=seed, n_jobs=-1,
        )
        self.feature_importances_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestIDS':
        print(f"  [Baseline] Training {self.NAME}...")
        self.clf.fit(X, y)
        self.feature_importances_ = self.clf.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        return _metrics(y, y_pred, y_prob, self.NAME)


# ═══════════════════════════════════════════════════════════════════════
class DecisionTreeIDS:
    NAME = 'Decision Tree IDS'

    def __init__(self, max_depth: int = 15, seed: int = 42):
        self.clf = DecisionTreeClassifier(
            max_depth=max_depth, class_weight='balanced', random_state=seed,
        )
        self.feature_importances_: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeIDS':
        print(f"  [Baseline] Training {self.NAME}...")
        self.clf.fit(X, y)
        self.feature_importances_ = self.clf.feature_importances_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = self.clf.predict_proba(X)
        return prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        return _metrics(y, y_pred, y_prob, self.NAME)


# ═══════════════════════════════════════════════════════════════════════
class ThresholdIDS:
    """
    Rule-based anomaly detector.
    Uses a statistical Z-score threshold on flow bytes/s and packets/s.
    Represents a traditional signature/rule-based IDS (e.g. Snort-style).
    """
    NAME = 'Threshold IDS (Rule-based)'

    def __init__(self, z_threshold: float = 2.5):
        self.z_thresh = z_threshold
        self._mean: np.ndarray = np.array([])
        self._std:  np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ThresholdIDS':
        print(f"  [Baseline] Training {self.NAME}...")
        benign = X[y == 0]
        self._mean = benign.mean(axis=0)
        self._std  = benign.std(axis=0) + 1e-8
        return self

    def _anomaly_score(self, X: np.ndarray) -> np.ndarray:
        z = np.abs((X - self._mean) / self._std)
        return z.max(axis=1)     # max Z-score across features

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._anomaly_score(X) > self.z_thresh).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._anomaly_score(X)
        return np.clip(scores / (self.z_thresh * 3), 0, 1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        return _metrics(y, y_pred, y_prob, self.NAME)


# ═══════════════════════════════════════════════════════════════════════
class CAESARAdaptiveIDS:
    """
    Wraps the CAESAR defender as an IDS classifier for fair comparison.

    The CAESAR ADPN doesn't directly output class labels — it outputs
    defense actions. This wrapper maps the Q-value signal to a binary
    detection score: high Q-value for any non-zero action → attack detected.
    """
    NAME = 'CAESAR Adaptive IDS'

    def __init__(self, adpn, env):
        self.adpn = adpn
        self.env  = env

    def _state_from_row(self, row: np.ndarray) -> np.ndarray:
        """Map a dataset feature row to an env-compatible state vector."""
        state = np.zeros(self.env.STATE_DIM, dtype=np.float32)
        n = min(len(row), self.env.STATE_DIM)
        state[:n] = row[:n]
        return state

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(X), dtype=np.float32)
        for i, row in enumerate(X):
            state = self._state_from_row(row)
            q     = self.adpn.q_values(state)
            # Probability of attack = 1 - P(no-action)
            q_soft = np.exp(q - q.max())
            q_soft /= q_soft.sum()
            scores[i] = 1.0 - q_soft[0]   # action 0 = NO_ACTION
        return scores

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= 0.5).astype(int)
        return _metrics(y, y_pred, y_prob, self.NAME)


# ═══════════════════════════════════════════════════════════════════════
#  State-of-the-Art IDS Comparison Baselines
# ═══════════════════════════════════════════════════════════════════════

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class IDSGANBaseline:
    """
    Simulates the IDSGAN adversarial evasion approach (Lin et al., 2022).

    IDSGAN trains a GAN where the generator learns to modify malicious
    traffic features so that a black-box IDS misclassifies attacks as
    benign.  This baseline approximates that pipeline with a lightweight
    autoencoder-based perturbation model (no PyTorch required):

      1. A Random Forest detector is trained on the original data.
      2. An autoencoder (single hidden-layer, numpy-only) is fitted on
         attack samples to learn a compressed representation.
      3. At evaluation time, attack samples are reconstructed through the
         autoencoder and small noise is added to simulate adversarial
         perturbation.
      4. The evasion_rate measures what fraction of true attacks are
         misclassified as benign after perturbation.

    Interface
    ---------
    .fit(X_train, y_train)
    .evaluate(X_test, y_test) → Dict  (includes 'evasion_rate')
    """

    NAME = 'IDSGAN Baseline (Lin et al., 2022)'

    def __init__(self, hidden_dim: int = 32, noise_scale: float = 0.15,
                 n_estimators: int = 100, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.noise_scale = noise_scale
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.detector = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=20,
            class_weight='balanced', random_state=seed, n_jobs=-1,
        )
        # Autoencoder weights (set during fit)
        self._W_enc: np.ndarray | None = None
        self._b_enc: np.ndarray | None = None
        self._W_dec: np.ndarray | None = None
        self._b_dec: np.ndarray | None = None

    # ── simple numpy autoencoder ──────────────────────────────────────
    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _init_weights(self, input_dim: int) -> None:
        scale = np.sqrt(2.0 / input_dim)
        self._W_enc = self.rng.randn(input_dim, self.hidden_dim).astype(np.float32) * scale
        self._b_enc = np.zeros(self.hidden_dim, dtype=np.float32)
        self._W_dec = self.rng.randn(self.hidden_dim, input_dim).astype(np.float32) * scale
        self._b_dec = np.zeros(input_dim, dtype=np.float32)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        return self._relu(X @ self._W_enc + self._b_enc)

    def _decode(self, H: np.ndarray) -> np.ndarray:
        return self._sigmoid(H @ self._W_dec + self._b_dec)

    def _train_autoencoder(self, X_attack: np.ndarray,
                           epochs: int = 50, lr: float = 0.01,
                           batch_size: int = 128) -> None:
        """Train a one-hidden-layer autoencoder with SGD (numpy only)."""
        self._init_weights(X_attack.shape[1])
        n = len(X_attack)
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = X_attack[idx[start:start + batch_size]]
                # Forward
                h = self._encode(batch)
                x_hat = self._decode(h)
                # Loss gradient (MSE)
                diff = x_hat - batch                        # (B, D)
                grad_dec = h.T @ diff / len(batch)          # (H, D)
                grad_b_dec = diff.mean(axis=0)
                d_h = diff @ self._W_dec.T * (h > 0)       # relu derivative
                grad_enc = batch.T @ d_h / len(batch)
                grad_b_enc = d_h.mean(axis=0)
                # Update
                self._W_dec -= lr * grad_dec
                self._b_dec -= lr * grad_b_dec
                self._W_enc -= lr * grad_enc
                self._b_enc -= lr * grad_b_enc

    def _perturb(self, X_attack: np.ndarray) -> np.ndarray:
        """Generate adversarial variants of attack samples."""
        h = self._encode(X_attack)
        x_recon = self._decode(h)
        noise = self.rng.randn(*X_attack.shape).astype(np.float32) * self.noise_scale
        return np.clip(x_recon + noise, 0, 1)

    # ── public API ────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'IDSGANBaseline':
        """Train the black-box detector and the adversarial evasion model."""
        print(f"  [Baseline] Training {self.NAME}...")
        self.detector.fit(X, y)
        X_attack = X[y == 1]
        if len(X_attack) > 0:
            self._train_autoencoder(X_attack.astype(np.float32))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.detector.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.detector.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate detection performance and adversarial evasion rate.

        Returns standard metrics plus *evasion_rate*: fraction of true
        attacks that evade the detector after adversarial perturbation.
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        results = _metrics(y, y_pred, y_prob, self.NAME)

        # Compute evasion rate on attack samples
        X_attack = X[y == 1]
        if len(X_attack) > 0 and self._W_enc is not None:
            X_adv = self._perturb(X_attack.astype(np.float32))
            preds_adv = self.detector.predict(X_adv)
            evasion_rate = float((preds_adv == 0).sum()) / len(X_attack)
        else:
            evasion_rate = 0.0
        results['evasion_rate'] = evasion_rate
        return results


# ═══════════════════════════════════════════════════════════════════════
class WGANIDSBaseline:
    """
    Simulates WGAN-based IDS data augmentation (Ring et al., 2019).

    Wasserstein GAN augmentation trains a generator to produce realistic
    synthetic attack samples, which are mixed into the training set to
    improve classifier robustness against under-represented attack types.

    This baseline approximates the generator with a simple noise-to-sample
    mapping learned via a one-hidden-layer network (numpy only):

      1. A generator network maps random noise → synthetic attack features.
      2. The generator is trained to minimise reconstruction error against
         real attack samples (lightweight surrogate for the WGAN objective).
      3. Synthetic attacks are appended to the training set and a Random
         Forest classifier is trained on the augmented data.
      4. *augmented_accuracy* reports accuracy of the augmented model vs.
         the non-augmented baseline.

    Interface
    ---------
    .fit(X_train, y_train)
    .evaluate(X_test, y_test) → Dict  (includes 'augmented_accuracy')
    """

    NAME = 'WGAN-IDS Baseline (Ring et al., 2019)'

    def __init__(self, n_synthetic: int = 500, hidden_dim: int = 64,
                 n_estimators: int = 100, seed: int = 42):
        self.n_synthetic = n_synthetic
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=20,
            class_weight='balanced', random_state=seed, n_jobs=-1,
        )
        self.clf_base = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=20,
            class_weight='balanced', random_state=seed, n_jobs=-1,
        )
        # Generator weights
        self._W1: np.ndarray | None = None
        self._b1: np.ndarray | None = None
        self._W2: np.ndarray | None = None
        self._b2: np.ndarray | None = None

    def _train_generator(self, X_attack: np.ndarray,
                         epochs: int = 80, lr: float = 0.005,
                         batch_size: int = 64) -> None:
        """Train a simple generator: noise → synthetic attack sample."""
        input_dim = X_attack.shape[1]
        scale = np.sqrt(2.0 / input_dim)
        self._W1 = self.rng.randn(input_dim, self.hidden_dim).astype(np.float32) * scale
        self._b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self._W2 = self.rng.randn(self.hidden_dim, input_dim).astype(np.float32) * scale
        self._b2 = np.zeros(input_dim, dtype=np.float32)

        n = len(X_attack)
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                real = X_attack[idx[start:start + batch_size]]
                z = self.rng.randn(len(real), input_dim).astype(np.float32)
                # Forward
                h = np.maximum(0, z @ self._W1 + self._b1)
                x_gen = h @ self._W2 + self._b2
                # MSE loss gradient
                diff = x_gen - real
                grad_W2 = h.T @ diff / len(real)
                grad_b2 = diff.mean(axis=0)
                d_h = diff @ self._W2.T * (h > 0)
                grad_W1 = z.T @ d_h / len(real)
                grad_b1 = d_h.mean(axis=0)
                # Update
                self._W2 -= lr * grad_W2
                self._b2 -= lr * grad_b2
                self._W1 -= lr * grad_W1
                self._b1 -= lr * grad_b1

    def _generate(self, n_samples: int, input_dim: int) -> np.ndarray:
        """Generate synthetic attack samples from random noise."""
        z = self.rng.randn(n_samples, input_dim).astype(np.float32)
        h = np.maximum(0, z @ self._W1 + self._b1)
        return h @ self._W2 + self._b2

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WGANIDSBaseline':
        """Train the generator, augment data, and train the RF classifier."""
        print(f"  [Baseline] Training {self.NAME}...")
        # Train base (non-augmented) classifier
        self.clf_base.fit(X, y)

        X_attack = X[y == 1]
        if len(X_attack) > 0:
            self._train_generator(X_attack.astype(np.float32))
            X_syn = self._generate(self.n_synthetic, X.shape[1])
            y_syn = np.ones(self.n_synthetic, dtype=y.dtype)
            X_aug = np.vstack([X, X_syn])
            y_aug = np.concatenate([y, y_syn])
        else:
            X_aug, y_aug = X, y

        self.clf.fit(X_aug, y_aug)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate augmented classifier and report improvement over baseline.

        Returns standard metrics plus *augmented_accuracy*: accuracy of the
        WGAN-augmented model (vs. *accuracy* which is the same value, kept
        for consistency with the base RF result available via clf_base).
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        results = _metrics(y, y_pred, y_prob, self.NAME)
        results['augmented_accuracy'] = results['accuracy']
        # Also report base (non-augmented) accuracy for comparison
        y_pred_base = self.clf_base.predict(X)
        results['base_accuracy'] = float(accuracy_score(y, y_pred_base))
        return results


# ═══════════════════════════════════════════════════════════════════════
class DeepRLDefender:
    """
    Simulates a basic Deep Reinforcement Learning IDS defence agent.

    Unlike CAESAR's co-evolutionary Dueling-Double-DQN with TAG/TA-GAN,
    this baseline uses simple tabular Q-learning to select defence actions
    based on discretised network state observations.  It serves as a
    non-co-evolutionary DRL reference point.

    States are discretised by binning each feature into *n_bins* levels.
    Actions: 0 = allow, 1 = block.

    Interface
    ---------
    .fit(X_train, y_train)
    .evaluate(X_test, y_test) → Dict  (includes 'avg_reward')
    """

    NAME = 'Deep RL Defender (Tabular Q-Learning)'

    def __init__(self, n_bins: int = 10, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 0.1,
                 n_episodes: int = 5, seed: int = 42):
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.q_table: Dict[tuple, np.ndarray] = {}
        self._bin_edges: list | None = None

    def _discretise(self, X: np.ndarray) -> np.ndarray:
        """Bin continuous features into integer indices."""
        disc = np.zeros_like(X, dtype=int)
        for j in range(X.shape[1]):
            disc[:, j] = np.digitize(X[:, j], self._bin_edges[j]) - 1
            disc[:, j] = np.clip(disc[:, j], 0, self.n_bins - 1)
        return disc

    def _state_key(self, row: np.ndarray) -> tuple:
        return tuple(row.tolist())

    def _get_q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(2, dtype=np.float64)
        return self.q_table[state]

    def _choose_action(self, state: tuple) -> int:
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(2))
        q = self._get_q(state)
        return int(np.argmax(q))

    @staticmethod
    def _reward(action: int, label: int) -> float:
        """
        +1  correct block (attack blocked)
        +0.5 correct allow (benign allowed)
        -1  missed attack (false negative)
        -0.5 false alarm (false positive)
        """
        if label == 1 and action == 1:
            return 1.0
        if label == 0 and action == 0:
            return 0.5
        if label == 1 and action == 0:
            return -1.0
        return -0.5  # label == 0, action == 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DeepRLDefender':
        """Train the Q-learning agent over multiple episodes."""
        print(f"  [Baseline] Training {self.NAME}...")
        # Compute bin edges from training data
        self._bin_edges = []
        for j in range(X.shape[1]):
            edges = np.linspace(X[:, j].min(), X[:, j].max(), self.n_bins + 1)[1:-1]
            self._bin_edges.append(edges)

        X_disc = self._discretise(X)

        for _ep in range(self.n_episodes):
            idx = self.rng.permutation(len(X_disc))
            for i in range(len(idx) - 1):
                s = self._state_key(X_disc[idx[i]])
                a = self._choose_action(s)
                r = self._reward(a, int(y[idx[i]]))
                s_next = self._state_key(X_disc[idx[i + 1]])
                # Q-learning update
                q = self._get_q(s)
                q_next = self._get_q(s_next)
                q[a] += self.alpha * (r + self.gamma * q_next.max() - q[a])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_disc = self._discretise(X)
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(X_disc)):
            s = self._state_key(X_disc[i])
            q = self._get_q(s)
            preds[i] = int(np.argmax(q))
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return soft 'attack probability' from Q-values."""
        X_disc = self._discretise(X)
        probs = np.zeros(len(X), dtype=np.float64)
        for i in range(len(X_disc)):
            s = self._state_key(X_disc[i])
            q = self._get_q(s)
            exp_q = np.exp(q - q.max())
            probs[i] = exp_q[1] / exp_q.sum()  # softmax P(block)
        return probs

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate detection performance and average reward.

        Returns standard metrics plus *avg_reward*: mean per-sample reward
        achieved by the learned policy on the test set.
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        results = _metrics(y, y_pred, y_prob, self.NAME)

        # Compute average reward
        rewards = np.array([self._reward(int(p), int(t))
                            for p, t in zip(y_pred, y)])
        results['avg_reward'] = float(rewards.mean())
        return results


# ═══════════════════════════════════════════════════════════════════════
class SOTAComparison:
    """
    Utility class for running all IDS baselines and producing comparison
    tables and figures suitable for a PhD thesis.

    Methods
    -------
    compare_all(X_train, y_train, X_test, y_test) → pd.DataFrame
        Trains every baseline on the training split and evaluates on the
        test split.  Returns a DataFrame with one row per baseline.

    plot_comparison(results, save_path)
        Generates a grouped bar chart comparing key metrics across all
        baselines and saves it to *save_path*.

    generate_latex_table(results) → str
        Returns a LaTeX-formatted table string ready for inclusion in a
        thesis document.
    """

    # Default set of baselines (can be overridden)
    BASELINES = [
        RandomForestIDS,
        DecisionTreeIDS,
        ThresholdIDS,
        IDSGANBaseline,
        WGANIDSBaseline,
        DeepRLDefender,
    ]

    DISPLAY_METRICS = ['accuracy', 'f1', 'detection_rate', 'false_pos_rate', 'roc_auc']

    def __init__(self, baselines: list | None = None):
        if baselines is not None:
            self.BASELINES = baselines

    def compare_all(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Train and evaluate all baselines.

        Returns a pandas DataFrame with one row per baseline and columns
        for every metric returned by evaluate().
        """
        rows = []
        for cls in self.BASELINES:
            model = cls()
            model.fit(X_train, y_train)
            metrics = model.evaluate(X_test, y_test)
            rows.append(metrics)
        return pd.DataFrame(rows)

    @staticmethod
    def plot_comparison(results: pd.DataFrame,
                        save_path: str = 'baseline_comparison.png') -> None:
        """Generate a grouped bar chart comparing key metrics."""
        metrics = [m for m in SOTAComparison.DISPLAY_METRICS if m in results.columns]
        models = results['model'].tolist()
        n_models = len(models)
        n_metrics = len(metrics)

        x = np.arange(n_metrics)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, model_name in enumerate(models):
            row = results[results['model'] == model_name].iloc[0]
            vals = [row[m] for m in metrics]
            ax.bar(x + i * width, vals, width, label=model_name)

        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics],
                           rotation=20, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('CAESAR — IDS Baseline Comparison')
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"  [SOTAComparison] Saved comparison chart → {save_path}")

    @staticmethod
    def generate_latex_table(results: pd.DataFrame) -> str:
        """
        Return a LaTeX booktabs table string for thesis inclusion.

        Example output (abbreviated)::

            \\begin{table}[ht]
            \\centering
            \\caption{IDS Baseline Comparison}
            \\begin{tabular}{lrrrrr}
            \\toprule
            Model & Accuracy & F1 & Det. Rate & FPR & AUC \\\\
            \\midrule
            Random Forest IDS & 0.95 & 0.93 & ... \\\\
            ...
            \\bottomrule
            \\end{tabular}
            \\end{table}
        """
        metrics = [m for m in SOTAComparison.DISPLAY_METRICS if m in results.columns]
        col_headers = {
            'accuracy': 'Accuracy',
            'f1': 'F1',
            'detection_rate': 'Det.\\ Rate',
            'false_pos_rate': 'FPR',
            'roc_auc': 'AUC',
        }

        header_line = 'Model & ' + ' & '.join(col_headers.get(m, m) for m in metrics) + ' \\\\'
        lines = [
            '\\begin{table}[ht]',
            '\\centering',
            '\\caption{IDS Baseline Comparison}',
            '\\begin{tabular}{l' + 'r' * len(metrics) + '}',
            '\\toprule',
            header_line,
            '\\midrule',
        ]

        for _, row in results.iterrows():
            vals = ' & '.join(f'{row[m]:.4f}' for m in metrics)
            lines.append(f'{row["model"]} & {vals} \\\\')

        lines += [
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}',
        ]
        return '\n'.join(lines)
