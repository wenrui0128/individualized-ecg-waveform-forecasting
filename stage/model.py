"""Source-conditioned forward ECG forecasting; one progression method only."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import torch
from torch import nn
from .components import (
    AdaptiveIntensityNet, DisentangledBeatVAE,
    GaussianPrototypeMemory, memory_global_aging_vector,
)


@dataclass(frozen=True)
class ModelConfig:
    window: int = 512
    in_channels: int = 1
    z_evo_dim: int = 8
    z_inv_dim: int = 40
    age_min: int = 18
    age_max: int = 95
    age_step: int = 1
    proto_sigma: float = 2.0
    vae_logvar_min: float = -8.0
    vae_logvar_max: float = 4.0
    adv_capacity_mult: float = 0.5
    adv_spectral_norm: bool = False
    adaptive_base_intensity: float = 3.0
    adaptive_min_mult: float = 0.4
    adaptive_max_mult: float = 1.8

    @property
    def z_dim(self):
        return self.z_evo_dim + self.z_inv_dim

    def __post_init__(self):
        if (self.window, self.in_channels, self.z_evo_dim, self.z_inv_dim) != (512, 1, 8, 40):
            raise ValueError('This release supports the fixed Lead-II 512 / 8+40 architecture.')
        if self.age_min >= self.age_max or self.age_step <= 0 or self.proto_sigma <= 0:
            raise ValueError('Invalid Gaussian age-memory configuration.')
        if not (0 < self.adaptive_min_mult < self.adaptive_max_mult and self.adaptive_base_intensity > 0):
            raise ValueError('Invalid intensity bounds.')


class STAGE(nn.Module):
    """VAE + Gaussian memory + Adaptive-global intensity, with no method switch.

    G0 (zero horizon) is exactly the input Source. Positive horizons decode a
    displaced source latent; the reconstruction is returned separately.
    """
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.backbone = DisentangledBeatVAE(c)
        self.proto = GaussianPrototypeMemory(c.age_min, c.age_max, c.age_step, c.z_evo_dim, c.proto_sigma)
        self.adaptive_intensity = AdaptiveIntensityNet(c)

    def forward(self, source, source_age, target_age, *, gain: float = 1.0):
        """Forecast [B,1,512] calibrated-mV beats, ages in years.

        Unlike the historical wrapper, out-of-range ages are rejected rather
        than silently clamped. Gain multiplies the learned intensity and must
        be declared in experiment reports. Use eval() for deterministic output.
        """
        c = self.config
        if source.ndim != 3 or tuple(source.shape[1:]) != (1, 512) or not source.is_floating_point():
            raise ValueError('source must be a floating tensor [B,1,512].')
        if source.shape[0] == 0 or not torch.isfinite(source).all():
            raise ValueError('source must be non-empty and finite.')
        a = torch.as_tensor(source_age, dtype=source.dtype, device=source.device).reshape(-1)
        b = torch.as_tensor(target_age, dtype=source.dtype, device=source.device).reshape(-1)
        if a.numel() != source.shape[0] or b.numel() != source.shape[0]:
            raise ValueError('Provide one source and target age per beat.')
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError('Ages must be finite.')
        if ((a < c.age_min) | (b > c.age_max) | (b < a)).any():
            raise ValueError(f'Require {c.age_min} <= source_age <= target_age <= {c.age_max}.')
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError('gain must be finite and positive.')
        enc = self.backbone.encode(source, alpha=0, sample=False, head_source='mu')
        evo, inv = enc['mu_evo'], enc['mu_inv']
        delta = b - a
        span = float(c.age_max - c.age_min)
        features = torch.cat((evo, torch.zeros_like(evo), inv,
                              ((a-c.age_min)/span)[:, None], ((b-c.age_min)/span)[:, None],
                              (delta/span)[:, None], (delta.abs()/span)[:, None], delta.sign()[:, None]), 1)
        adaptive = self.adaptive_intensity(features)
        intensity = adaptive['intensity'] * gain
        direction = memory_global_aging_vector(self.proto, c.age_min, c.age_max)[None]
        target_evo = evo + direction * delta[:, None] * intensity
        prediction = self.backbone.decode(target_evo, inv)
        prediction = torch.where((delta == 0)[:, None, None], source, prediction)
        return dict(forecast=prediction, reconstruction=self.backbone.decode(evo, inv),
                    source_evo=evo, target_evo=target_evo, invariant=inv,
                    intensity=intensity, multiplier=adaptive['multiplier'], direction=direction)

    def save(self, path, *, metadata=None):
        """Save only this method's tensors/configuration, without patient data."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(format='stage-adaptive-global-v1', config=asdict(self.config),
                        model={k: v.detach().cpu() for k, v in self.state_dict().items()},
                        metadata=metadata or {}), path)

    @classmethod
    def load(cls, path, device='cpu'):
        """Load restricted-format tensors; no original training-object unpickling."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('format') != 'stage-adaptive-global-v1':
            raise ValueError('Convert the original checkpoint with stage.export first.')
        model = cls(ModelConfig(**checkpoint['config']))
        model.load_state_dict(checkpoint['model'], strict=True)
        return model.to(device).eval()
