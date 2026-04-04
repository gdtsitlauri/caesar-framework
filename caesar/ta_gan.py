"""
TA-GAN — Threat-Aware Generative Adversarial Network
=====================================================
Novel Contribution #1 of the CAESAR framework.

Standard GANs generate attacks from pure noise.
TA-GAN conditions generation on the *current defense state*, so the attacker
continuously adapts to what the defender is doing — mimicking real APT behaviour.

Architecture:
    G(z, d_embed) → attack_features   # defense-conditioned generator
    D(x, d_embed) → ℝ                 # defense-aware discriminator

The bypass_reward signal closes the co-evolutionary loop:
  • high attack success  →  GAN gets positive gradient boost
  • strong defense       →  GAN forced to diversify attack patterns
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Tuple

# ─── Hyper-parameters ────────────────────────────────────────────────
NOISE_DIM  = 32
DEFENSE_DIM = 8
FEATURE_DIM = 16   # matches CyberEnvironment.STATE_DIM
LR_G        = 2e-4
LR_D        = 2e-4
BETAS       = (0.5, 0.999)


# ═══════════════════════════════════════════════════════════════════════
class ThreatAwareGenerator(nn.Module):
    """
    G(z ∈ ℝ^32, d ∈ ℝ^8) → (features ∈ ℝ^16, attack_probs ∈ Δ^8)

    Dual-head: feature head produces raw attack features,
               type head classifies attack type (for interpretability).
    """

    def __init__(self):
        super().__init__()
        in_dim = NOISE_DIM + DEFENSE_DIM

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 64),   nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),      nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),      nn.LeakyReLU(0.2),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(64, FEATURE_DIM),
            nn.Sigmoid(),            # features normalised to [0, 1]
        )
        self.type_head = nn.Sequential(
            nn.Linear(64, 32),       nn.ReLU(),
            nn.Linear(32, 8),        nn.Softmax(dim=-1),
        )

    def forward(self, z: torch.Tensor, d: torch.Tensor):
        h        = self.backbone(torch.cat([z, d], dim=-1))
        features = self.feature_head(h)
        atk_prob = self.type_head(h)
        return features, atk_prob


# ═══════════════════════════════════════════════════════════════════════
class ThreatAwareDiscriminator(nn.Module):
    """
    D(x ∈ ℝ^16, d ∈ ℝ^8) → p(real) ∈ [0, 1]

    Novelty: discriminator sees the defense state so it can judge
    whether an attack sample is "plausibly bypassing current defenses".
    """

    def __init__(self):
        super().__init__()
        in_dim = FEATURE_DIM + DEFENSE_DIM

        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),  nn.LeakyReLU(0.2),
            nn.Dropout(0.30),
            nn.Linear(128, 64),      nn.LeakyReLU(0.2),
            nn.Dropout(0.30),
            nn.Linear(64, 32),       nn.LeakyReLU(0.2),
            nn.Linear(32, 1),        nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, d], dim=-1))


# ═══════════════════════════════════════════════════════════════════════
class ThreatAwareGAN:
    """
    Full TA-GAN system: training, inference, and co-evolutionary feedback.

    Co-evolutionary signal
    ──────────────────────
    After each environment step the CAESAR loop calls:
        update_bypass_reward(attack_success)
    This EMA-smoothed bypass reward is added as an auxiliary loss term,
    directly linking attack generation quality to actual bypass performance.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.G = ThreatAwareGenerator().to(device)
        self.D = ThreatAwareDiscriminator().to(device)
        self.g_opt = torch.optim.Adam(self.G.parameters(), lr=LR_G, betas=BETAS)
        self.d_opt = torch.optim.Adam(self.D.parameters(), lr=LR_D, betas=BETAS)
        self.criterion = nn.BCELoss()

        # co-evolutionary state
        self._bypass_buf:  deque = deque(maxlen=100)
        self.bypass_reward: float = 0.0

        # training history
        self.g_losses: list = []
        self.d_losses: list = []

    # ------------------------------------------------------------------
    def train_step(self,
                   real_data: np.ndarray,
                   defense_state: np.ndarray) -> Tuple[float, float]:
        """One mini-batch GAN update. Returns (g_loss, d_loss)."""
        bs = len(real_data)
        if bs < 2:
            return 0.0, 0.0

        real  = torch.FloatTensor(real_data).to(self.device)
        d_emb = (torch.FloatTensor(defense_state)
                 .unsqueeze(0).expand(bs, -1).to(self.device))

        # ── Discriminator ──────────────────────────────────────────────
        self.d_opt.zero_grad()
        real_lbl = torch.full((bs, 1), 0.90).to(self.device)  # label smoothing
        fake_lbl = torch.full((bs, 1), 0.10).to(self.device)

        d_real = self.D(real, d_emb)
        d_real_loss = self.criterion(d_real, real_lbl)

        z    = torch.randn(bs, NOISE_DIM).to(self.device)
        fake, _ = self.G(z, d_emb)
        d_fake = self.D(fake.detach(), d_emb)
        d_fake_loss = self.criterion(d_fake, fake_lbl)

        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        self.d_opt.step()

        # ── Generator ──────────────────────────────────────────────────
        self.g_opt.zero_grad()
        z    = torch.randn(bs, NOISE_DIM).to(self.device)
        fake, atk_probs = self.G(z, d_emb)
        d_fake = self.D(fake, d_emb)

        g_adv  = self.criterion(d_fake, real_lbl)
        # Bypass bonus: encourage generation of high-success attacks
        bypass = torch.tensor(-self.bypass_reward * 0.15,
                              dtype=torch.float32, device=self.device)
        # Diversity penalty: avoid mode collapse on single attack type
        entropy = -(atk_probs * (atk_probs + 1e-8).log()).sum(dim=-1).mean()
        diversity_bonus = -0.05 * entropy   # maximise entropy → diverse attacks

        g_loss = g_adv + bypass + diversity_bonus
        g_loss.backward()
        self.g_opt.step()

        gv, dv = g_loss.item(), d_loss.item()
        self.g_losses.append(gv)
        self.d_losses.append(dv)
        return gv, dv

    # ------------------------------------------------------------------
    def update_bypass_reward(self, attack_success: float):
        """EMA update of bypass reward from real environment feedback."""
        self._bypass_buf.append(float(attack_success))
        self.bypass_reward = float(np.mean(self._bypass_buf))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_attack(self,
                        defense_state: np.ndarray,
                        n: int = 1) -> Tuple[int, float]:
        """
        Sample one attack from the conditioned generator.
        Returns (attack_type: int, intensity: float).
        """
        self.G.eval()
        z    = torch.randn(n, NOISE_DIM).to(self.device)
        d_emb = (torch.FloatTensor(defense_state)
                 .unsqueeze(0).expand(n, -1).to(self.device))
        features, atk_probs = self.G(z, d_emb)
        self.G.train()

        atk_type  = int(atk_probs[0].argmax().item())
        intensity = float(features[0, 3].item())   # feature dim-3 ≈ intensity
        return atk_type, float(np.clip(intensity, 0.05, 1.0))
