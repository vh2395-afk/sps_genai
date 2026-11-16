"""Diffusion model for Assignment 4 (CIFAR-10, 32x32x3).

- sinusoidal_embedding(t): timestep embedding (Part 2, Q1-Q3)
- UNet: predicts the noise eps added to x_t, conditioned on t
    32x32 -> 16x16 -> 8x8 -> 4x4 bottleneck (3 stride-2 downsamples), then back up with skips
- Diffusion: linear beta schedule (T=1000), forward noising q(x_t | x_0),
  training loss MSE(eps, eps_theta(x_t, t)), and DDIM sampling for fast generation.

Shape check:  uv run python -m app.diffusion
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 32
T_STEPS = 1000


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """emb[i] = sin(t * f_i), emb[half + i] = cos(t * f_i),  f_i = max_period^(-i / half)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]  # inject timestep
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, base: int = 64, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        t_dim = emb_dim * 2
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        c1, c2, c3 = base, base * 2, base * 2

        self.inc = nn.Conv2d(3, c1, 3, padding=1)
        self.enc1 = ResBlock(c1, c1, t_dim)   # 32x32
        self.down1 = Down(c1)                 # -> 16
        self.enc2 = ResBlock(c1, c2, t_dim)   # 16x16
        self.down2 = Down(c2)                 # -> 8
        self.enc3 = ResBlock(c2, c3, t_dim)   # 8x8
        self.down3 = Down(c3)                 # -> 4

        self.mid1 = ResBlock(c3, c3, t_dim)   # 4x4 bottleneck
        self.mid2 = ResBlock(c3, c3, t_dim)

        self.up3 = Up(c3)                     # -> 8
        self.dec3 = ResBlock(c3 + c3, c3, t_dim)
        self.up2 = Up(c3)                     # -> 16
        self.dec2 = ResBlock(c3 + c2, c2, t_dim)
        self.up1 = Up(c2)                     # -> 32
        self.dec1 = ResBlock(c2 + c1, c1, t_dim)

        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.time_mlp(sinusoidal_embedding(t, self.emb_dim))

        h = self.inc(x)
        s1 = self.enc1(h, te)
        s2 = self.enc2(self.down1(s1), te)
        s3 = self.enc3(self.down2(s2), te)

        m = self.mid2(self.mid1(self.down3(s3), te), te)

        d3 = self.dec3(torch.cat([self.up3(m), s3], dim=1), te)
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1), te)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1), te)
        return self.out(d1)  # predicted noise, same shape as x


class Diffusion:
    """DDPM noise schedule plus helpers for training and sampling."""

    def __init__(self, timesteps: int = T_STEPS, device: str | torch.device = "cpu"):
        self.T = timesteps
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: x_t = sqrt(a_bar_t) x_0 + sqrt(1 - a_bar_t) eps."""
        ab = self.alpha_bars[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        return F.mse_loss(model(xt, t), noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, steps: int = 50,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """DDIM (eta = 0) sampling with `steps` denoising steps. Returns images in [-1, 1]."""
        model.eval()
        x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=generator).to(self.device)
        times = torch.linspace(self.T - 1, 0, steps).long().tolist()
        for i, t in enumerate(times):
            ab = self.alpha_bars[t]
            ab_prev = self.alpha_bars[times[i + 1]] if i + 1 < len(times) else torch.tensor(1.0, device=self.device)
            t_batch = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = model(x, t_batch)
            x0_pred = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
        return x.clamp(-1, 1)


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    t = torch.randint(0, T_STEPS, (4,))
    print("embedding(t=1, dim=8):", sinusoidal_embedding(torch.tensor([1]), 8).numpy().round(4))
    print("UNet output:", tuple(model(x, t).shape))   # (4, 3, 32, 32)
    print("UNet params:", sum(p.numel() for p in model.parameters()))
    print("diffusion loss:", Diffusion().loss(model, x).item())
