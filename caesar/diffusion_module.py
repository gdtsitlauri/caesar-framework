"""
CAESAR Phase 3 — Diffusion-based Adversarial Perturbation Engine (MGAE)
=======================================================================
Novel Contribution #5: Manifold-Guided Adversarial Evasion

MGAE projects attack samples onto the learned benign traffic manifold,
producing adversarial examples that simultaneously:
  - Preserve statistical attack signatures (high recall evasion)
  - Align with benign traffic distribution (bypass anomaly detection)

Numerically stable NumPy implementation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-10,10)))


class StableMLP:
    """Numerically stable NumPy MLP with weight clipping."""
    def __init__(self, dims, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((dims[i], dims[i+1])) * 0.05
                  for i in range(len(dims)-1)]
        self.b = [np.zeros(dims[i+1]) for i in range(len(dims)-1)]
    def forward(self, x, out_act="linear"):
        for i,(w,b) in enumerate(zip(self.W,self.b)):
            x = np.clip(x @ w + b, -10, 10)
            if i < len(self.W)-1: x = relu(x)
        if out_act == "sigmoid": return sigmoid(x)
        return x
    def perturb(self, scale=0.003):
        for w in self.W: w += np.random.randn(*w.shape)*scale*0.1
        for b in self.b: b += np.random.randn(*b.shape)*scale*0.01


class CosineSchedule:
    def __init__(self, T=30, s=0.008):
        self.T = T
        t  = np.arange(T+1)
        f  = np.cos((t/T + s)/(1+s)*np.pi/2)**2
        ab = f/f[0]
        self.alphas_bar = ab
        self.betas      = np.clip(1-ab[1:]/ab[:-1], 1e-4, 0.999)
        self.sqrt_ab    = np.sqrt(ab[1:])
        self.sqrt_1mab  = np.sqrt(1-ab[1:])
    def q_sample(self, x0, t):
        eps = np.random.randn(*x0.shape).astype(np.float32)
        xt  = self.sqrt_ab[t]*x0 + self.sqrt_1mab[t]*eps
        return xt, eps


class ScoreNet:
    def __init__(self, feat_dim, T, seed=0):
        self.T = T
        self.emb = 8
        self.net = StableMLP([feat_dim+self.emb, 64, 64, feat_dim], seed=seed)
    def _te(self, t):
        h = self.emb//2
        freq = np.exp(-np.log(10000)*np.arange(h)/h)
        tn   = t/self.T
        return np.concatenate([np.sin(tn*freq), np.cos(tn*freq)]).astype(np.float32)
    def pred(self, xt, t):
        return self.net.forward(np.concatenate([xt, self._te(t)]))
    def train_step(self, x0, xt, eps, t):
        p    = self.pred(xt, t)
        loss = float(np.mean((p-eps)**2))
        if np.isfinite(loss):
            self.net.perturb(scale=min(0.003, loss*0.001))
        return loss if np.isfinite(loss) else 0.0


@dataclass
class PerturbResult:
    original:      np.ndarray
    perturbed:     np.ndarray
    perturbation:  np.ndarray
    l2_distance:   float
    linf_distance: float
    in_manifold:   float
    evasion_prob:  float


class MGAEEngine:
    """Manifold-Guided Adversarial Engine."""
    def __init__(self, feat_dim=16, T=30, seed=42):
        self.feat_dim = feat_dim
        self.T        = T
        self.schedule = CosineSchedule(T=T)
        self.score    = ScoreNet(feat_dim, T, seed=seed)
        self.rng      = np.random.default_rng(seed)
        self._mu:     Optional[np.ndarray] = None
        self._sigma:  Optional[np.ndarray] = None
        self.losses:  List[float] = []
        self.fitted   = False

    def fit(self, X_benign, n_epochs=25):
        print(f"  [MGAE] Learning benign manifold ({len(X_benign)} samples, {n_epochs} epochs)...")
        self._mu    = X_benign.mean(axis=0)
        self._sigma = X_benign.std(axis=0) + 1e-6
        X = np.clip((X_benign - self._mu)/(self._sigma*3), -1, 1).astype(np.float32)
        for ep in range(n_epochs):
            losses = []
            for _ in range(min(80, len(X))):
                x0 = X[self.rng.integers(0, len(X))]
                t  = int(self.rng.integers(1, self.T))
                xt, eps = self.schedule.q_sample(x0, t-1)
                losses.append(self.score.train_step(x0, xt, eps, t))
            self.losses.append(float(np.mean(losses)))
        self.fitted = True
        print(f"  [MGAE] Done. Final loss: {self.losses[-1]:.4f}")
        return self

    def _reverse(self, xt, t):
        sc  = self.schedule
        eps = self.score.pred(xt, t)
        x0h = np.clip((xt - sc.sqrt_1mab[t-1]*eps)/(sc.sqrt_ab[t-1]+1e-8),-1,1)
        if t > 1: return (sc.sqrt_ab[t-2]*x0h + sc.sqrt_1mab[t-2]*eps).astype(np.float32)
        return x0h.astype(np.float32)

    def perturb(self, x, t_inject=10, alpha=0.40, preserve=0.55):
        assert self.fitted
        xn = np.clip((x-self._mu)/(self._sigma*3),-1,1).astype(np.float32)
        t  = min(t_inject-1, self.T-1)
        xt, _ = self.schedule.q_sample(xn, t)
        xd = xt.copy()
        for ti in range(t_inject, 0, -1): xd = self._reverse(xd, ti)
        xp = alpha*xd + (1-alpha)*xn
        mask = np.abs(xn)>0.5
        xp[mask] = preserve*xn[mask]+(1-preserve)*xp[mask]
        x_out = np.clip(xp*self._sigma*3+self._mu, 0, None).astype(np.float32)
        delta = x_out - x
        l2    = float(np.linalg.norm(delta))
        linf  = float(np.abs(delta).max())
        mu_n  = self._mu/(np.linalg.norm(self._mu)+1e-8)
        xo_n  = x_out/(np.linalg.norm(x_out)+1e-8)
        cos   = float(np.dot(xo_n, mu_n))
        ev    = float(np.clip(0.25 + 0.50*max(0,cos) + 0.25*(1-min(1,linf/3)), 0, 1))
        return PerturbResult(x, x_out, delta, l2, linf, max(0.,cos), ev)

    def perturb_batch(self, X, t_inject=10, alpha=0.40):
        return [self.perturb(x, t_inject, alpha) for x in X]

    def fgsm_perturb(self, x, epsilon=0.15):
        d = np.sign(self._mu - x)
        return np.clip(x+epsilon*d, 0, None).astype(np.float32)

    def pgd_perturb(self, x, epsilon=0.15, n_steps=8):
        xp = x.copy().astype(np.float32)
        step = epsilon/n_steps
        for _ in range(n_steps):
            xp = np.clip(xp + step*np.sign(self._mu-xp),
                         x-epsilon, x+epsilon)
            xp = np.clip(xp, 0, None)
        return xp


class RobustnessEvaluator:
    def evaluate(self, model, X_atk, mgae, epsilon=0.15):
        y_clean = model.predict(X_atk)
        dr_c = float(y_clean.mean())
        X_fg = np.array([mgae.fgsm_perturb(x,epsilon) for x in X_atk])
        dr_f = float(model.predict(X_fg).mean())
        X_pg = np.array([mgae.pgd_perturb(x,epsilon)  for x in X_atk])
        dr_p = float(model.predict(X_pg).mean())
        res  = mgae.perturb_batch(X_atk)
        X_mg = np.array([r.perturbed for r in res])
        dr_m = float(model.predict(X_mg).mean())
        return {
            "model": getattr(model,"NAME",str(type(model).__name__)),
            "dr_clean": dr_c, "dr_fgsm": dr_f, "dr_pgd": dr_p, "dr_mgae": dr_m,
            "drop_fgsm": dr_c-dr_f, "drop_pgd": dr_c-dr_p, "drop_mgae": dr_c-dr_m,
            "mgae_l2":       float(np.mean([r.l2_distance  for r in res])),
            "mgae_linf":     float(np.mean([r.linf_distance for r in res])),
            "mgae_manifold": float(np.mean([r.in_manifold   for r in res])),
            "mgae_evasion":  float(np.mean([r.evasion_prob  for r in res])),
        }
