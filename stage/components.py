"""Core components extracted from STAGE main-ii-v1 (see NOTICE.md).

Only components required by Adaptive-global are retained. Constructors use
instance configuration instead of the original process-global configuration.
Backbone topology derives from konspatl/vae_scan; see NOTICE.md.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class GradientReversal(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.alpha, None)

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

def _build_v10_inv_adversary(z_inv_dim: int, config) -> nn.Sequential:
    mult = float(getattr(config, 'adv_capacity_mult', 1.0))
    sn_enabled = bool(getattr(config, 'adv_spectral_norm', False))

    def _lin(in_f: int, out_f: int) -> nn.Module:
        layer = nn.Linear(in_f, out_f)
        if sn_enabled:
            layer = nn.utils.spectral_norm(layer)
        return layer
    if mult <= 0.0:
        return nn.Sequential(_lin(z_inv_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.3), _lin(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), _lin(128, 1))
    h1 = max(64, int(round(512 * mult)))
    h2 = max(64, int(round(512 * mult)))
    h3 = max(32, int(round(256 * mult)))
    h4 = max(32, int(round(128 * mult)))
    return nn.Sequential(_lin(z_inv_dim, h1), nn.BatchNorm1d(h1), nn.LeakyReLU(0.2), nn.Dropout(0.2), _lin(h1, h2), nn.BatchNorm1d(h2), nn.LeakyReLU(0.2), nn.Dropout(0.2), _lin(h2, h3), nn.BatchNorm1d(h3), nn.LeakyReLU(0.2), nn.Dropout(0.2), _lin(h3, h4), nn.BatchNorm1d(h4), nn.LeakyReLU(0.2), _lin(h4, 1))

class VaeScanConvBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, kernel: int, *, pool: bool=False):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.activation(self.conv(x)))

class VaeScanDecoderBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, kernel: int, *, upsample: bool=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else nn.Identity()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(self.upsample(x)))

class DisentangledBeatVAE(nn.Module):
    architecture_source = 'konspatl/vae_scan:vaescan/models.py simple_cnn_*_beat'

    def __init__(self, config):
        super().__init__()
        self.config = config
        if int(self.config.window) != 512 or int(self.config.in_channels) != 1:
            raise ValueError('VAE-SCAN II backbone requires config.window=512 and in_channels=1')
        self.encoder_blocks = nn.Sequential(VaeScanConvBlock(1, 32, 5), VaeScanConvBlock(32, 32, 5, pool=True), VaeScanConvBlock(32, 64, 3), VaeScanConvBlock(64, 128, 5, pool=True), VaeScanConvBlock(128, 256, 3), VaeScanConvBlock(256, 512, 5, pool=True), VaeScanConvBlock(512, 128, 3))
        self.last_channel = 128
        self.last_len = 64
        self.flatten_dim = self.last_channel * self.last_len
        self.fc_mu = nn.Linear(self.flatten_dim, self.config.z_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.config.z_dim)
        self.decoder_expand = nn.Linear(self.config.z_dim, self.flatten_dim)
        self.decoder_blocks = nn.Sequential(VaeScanDecoderBlock(128, 128, 3), VaeScanDecoderBlock(128, 512, 5, upsample=True), VaeScanDecoderBlock(512, 256, 3), VaeScanDecoderBlock(256, 128, 5, upsample=True), VaeScanDecoderBlock(128, 64, 3), VaeScanDecoderBlock(64, 32, 5, upsample=True), VaeScanDecoderBlock(32, 32, 5))
        self.final_conv = nn.Conv1d(32, self.config.in_channels, 5, padding=2)
        self.age_regressor = nn.Sequential(nn.Linear(self.config.z_evo_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.2), nn.Linear(64, 1))
        self.inv_adversary = _build_v10_inv_adversary(self.config.z_inv_dim, self.config)
        self.apply(self._init_reference_weights)

    @staticmethod
    def _init_reference_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, x: torch.Tensor, alpha: float=1.0, sample: bool=True, head_source: str='mu') -> dict[str, torch.Tensor]:
        if x.ndim != 3 or tuple(x.shape[1:]) != (1, self.config.window):
            raise ValueError(f'VAE-SCAN II encoder expects [B,1,{self.config.window}], got {tuple(x.shape)}')
        h = self.encoder_blocks(x).flatten(start_dim=1)
        if h.shape[1] != self.flatten_dim:
            raise RuntimeError(f'unexpected encoder flatten width {h.shape[1]} != {self.flatten_dim}')
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=float(self.config.vae_logvar_min), max=float(self.config.vae_logvar_max))
        z = self.reparameterize(mu, logvar) if sample else mu
        mu_evo = mu[:, :self.config.z_evo_dim]
        mu_inv = mu[:, self.config.z_evo_dim:]
        z_evo = z[:, :self.config.z_evo_dim]
        z_inv = z[:, self.config.z_evo_dim:]
        if head_source == 'z':
            age_head_in, adv_head_in = (z_evo, z_inv)
        elif head_source == 'mu':
            age_head_in, adv_head_in = (mu_evo, mu_inv)
        else:
            raise ValueError(f"head_source must be 'mu' or 'z', got {head_source!r}")
        return {'mu': mu, 'logvar': logvar, 'z': z, 'mu_evo': mu_evo, 'mu_inv': mu_inv, 'z_evo': z_evo, 'z_inv': z_inv, 'pred_age_main': self.age_regressor(age_head_in), 'pred_age_adv': self.inv_adversary(grad_reverse(adv_head_in, alpha))}

    def decode(self, z_evo: torch.Tensor, z_inv: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_evo, z_inv], dim=1)
        h = F.relu(self.decoder_expand(z))
        h = h.view(-1, self.last_channel, self.last_len)
        h = self.decoder_blocks(h)
        recon = self.final_conv(h)
        if recon.shape[-1] != self.config.window:
            raise RuntimeError(f'VAE-SCAN decoder produced {recon.shape[-1]} samples, expected {self.config.window}')
        return recon

    def forward(self, x: torch.Tensor, alpha: float=1.0, sample: bool=True, head_source: str='mu') -> dict[str, torch.Tensor]:
        enc = self.encode(x, alpha=alpha, sample=sample, head_source=head_source)
        enc['recon'] = self.decode(enc['z_evo'], enc['z_inv'])
        return enc

class GaussianPrototypeMemory(nn.Module):

    def __init__(self, age_min, age_max, age_step, z_evo_dim, sigma):
        super().__init__()
        anchors = torch.arange(age_min, age_max + age_step, age_step, dtype=torch.float32)
        self.register_buffer('age_anchors', anchors)
        self.sigma = float(sigma)
        self.logvar_min = -6.0
        self.logvar_max = 2.0
        self.memory_mu = nn.Parameter(0.01 * torch.randn(len(anchors), z_evo_dim))
        self.memory_logvar = nn.Parameter(torch.full((len(anchors), z_evo_dim), -2.0))

    def gaussian_weights(self, ages):
        ages = ages.view(-1, 1).float()
        dist2 = (ages - self.age_anchors.view(1, -1)).pow(2)
        w = torch.exp(-dist2 / (2.0 * self.sigma * self.sigma))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-08)
        return w

    def query(self, ages):
        w = self.gaussian_weights(ages)
        anchor_logvar = self.memory_logvar.clamp(min=self.logvar_min, max=self.logvar_max)
        anchor_var = anchor_logvar.exp()
        mu = w @ self.memory_mu
        second = w @ (anchor_var + self.memory_mu.square())
        var = (second - mu.square()).clamp(min=math.exp(self.logvar_min), max=math.exp(self.logvar_max))
        logvar = var.log()
        return {'mu': mu, 'logvar': logvar, 'w': w}

def memory_global_aging_vector(proto: GaussianPrototypeMemory, age_min=None, age_max=None) -> torch.Tensor:
    anchors = proto.age_anchors.float()
    mu = proto.memory_mu
    keep = torch.ones_like(anchors, dtype=torch.bool)
    if age_min is not None:
        keep &= anchors >= float(age_min)
    if age_max is not None:
        keep &= anchors <= float(age_max)
    anchors = anchors[keep]
    mu = mu[keep]
    if anchors.numel() < 2:
        return torch.zeros(mu.shape[-1], device=mu.device, dtype=mu.dtype)
    x = anchors.view(-1, 1).to(device=mu.device, dtype=mu.dtype)
    xc = x - x.mean(dim=0, keepdim=True)
    yc = mu - mu.mean(dim=0, keepdim=True)
    return (xc * yc).sum(dim=0) / (xc.pow(2).sum() + 1e-08)

class AdaptiveIntensityNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        feature_dim = self.config.z_evo_dim * 2 + self.config.z_inv_dim + 5
        self.net = nn.Sequential(nn.Linear(feature_dim, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(0.05), nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, subject_features: torch.Tensor) -> dict:
        raw = self.net(subject_features).view(-1, 1)
        min_mult = float(self.config.adaptive_min_mult)
        max_mult = float(self.config.adaptive_max_mult)
        if max_mult <= min_mult:
            max_mult = min_mult + 0.001
        mult = min_mult + (max_mult - min_mult) * torch.sigmoid(raw)
        base = float(self.config.adaptive_base_intensity if self.config.adaptive_base_intensity is not None else self.config.proto_intensity)
        intensity = base * mult
        return {'intensity': intensity, 'multiplier': mult, 'raw': raw}
