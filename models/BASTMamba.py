"""BAST-Mamba V3: two experimental changes to the supplied V2 model.

1. Fine-anchor residual fusion with token-dependent scale selection.
2. Input-dependent channel gates per sample, variable, and token.

Complete, standalone PyTorch model; input [B,L,C] -> output [B,T,C].
No measured accuracy improvement is claimed. Train and compare on your data.
The history normalization, BAST, FFT conditioner, single temporal backbone,
forecast head, and centered linear shortcut retain the V2 design.

Usage (CUDA with a compatible mamba_ssm installation):
    model = Model(Config(seq_len=96, pred_len=192, enc_in=21)).cuda()
    prediction = model(torch.randn(2, 96, 21, device="cuda"))
    loss = F.mse_loss(prediction, target)  # target: [2,192,21]

CPU functional tests (explicit convolution variant, NOT a Mamba emulation):
    python bast_mamba_v3.py --self-test
    python bast_mamba_v3.py --backend conv --smoke-test
CUDA forward/backward smoke test:
    python bast_mamba_v3.py --backend mamba --smoke-test

Two-by-two ablation, with all other training settings fixed:
    fusion_mode="global",       channel_gate=False  # V2 architecture
    fusion_mode="anchor_local", channel_gate=False  # change 1 only
    fusion_mode="global",       channel_gate=True   # change 2 only
    fusion_mode="anchor_local", channel_gate=True   # both, V3 default

Train each variant separately; V3 default checkpoints have additional keys.
Repeated patch sizes are retained as independent stems, unlike V2's set().
x_mark and y_true are accepted for interface compatibility and never used.
No training loop, data loader, or external dataset standardizer is assumed.
Python >= 3.10 and PyTorch >= 2.1 are recommended for this implementation.
Official Mamba source/API: https://github.com/state-spaces/mamba
"""

from dataclasses import dataclass
from typing import Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    seq_len: int
    pred_len: int
    enc_in: int
    d_model: int = 128
    patch_sizes: Tuple[int, ...] = (8, 16, 32)
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.1
    spectral_dim: int = 16
    local_kernel: int = 3
    head_rank: int = 64
    temporal_backend: str = "mamba"
    bast_gate_type: str = "mlp"
    bast_gate_hidden: int = 8
    bast_gate_init: float = 0.2
    bast_use_relative_jump: bool = True
    use_spectral: bool = True
    adaptive_scale: bool = True
    channel_mixing: bool = True
    channel_dim: int = 32
    num_routers: int = 4
    linear_skip: bool = True
    fusion_mode: str = "anchor_local"  # global / anchor_local
    fusion_hidden: int = 16
    fusion_residual_init: float = 0.1  # sigmoid-bounded coarse correction
    channel_gate: bool = True
    channel_gate_init: float = 0.2
    eps: float = 1e-5


def _get(configs, name, default=None):
    return configs.get(name, default) if isinstance(configs, dict) else getattr(configs, name, default)


def _boolean(value, name):
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{name} must be a boolean (not {value!r})")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fp_stats(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _logit(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return math.log(value / (1 - value))


class BoundaryTokenizer(nn.Module):
    """Fixed-length BAST; cached masks; optional local-shape descriptors.

    Statistics are masked for partial base patches. An incomplete shifted view
    is NEVER fused. The original half-patch shifted support is retained.
    """

    def __init__(self, length, patch, gate_type="mlp", hidden=8,
                 gate_init=0.2, relative=True, eps=1e-5):
        super().__init__()
        self.length = _positive(length, "length")
        self.patch = _positive(patch, "patch")
        self.shift = patch // 2
        self.count = math.ceil(length / patch)
        self.pad = self.count * patch - length
        if gate_type not in ("none", "fixed", "linear", "mlp"):
            raise ValueError("bast_gate_type must be none/fixed/linear/mlp")
        if not 0 < gate_init < 1 or eps <= 0:
            raise ValueError("Require 0 < bast_gate_init < 1 and eps > 0")
        self.kind, self.relative, self.eps = gate_type, relative, eps
        starts = torch.arange(self.count) * patch
        valid = (starts + self.shift + patch <= length) & (starts < (self.count - 1) * patch)
        if self.shift == 0:
            valid[:] = False
        self.register_buffer("valid", valid.reshape(1, 1, -1), persistent=False)
        observed = starts[:, None] + torch.arange(patch)[None, :] < length
        self.register_buffer("observed", observed.reshape(1, 1, self.count, patch), persistent=False)
        coverage = observed.sum(-1).float()
        self.register_buffer("pool_weight", (coverage / coverage.sum()).reshape(1, 1, -1, 1), persistent=False)
        feature_dim = 8 if relative else 5
        if gate_type in ("linear", "mlp") and self.shift:
            self.gate = (nn.Linear(feature_dim, 1) if gate_type == "linear" else
                         nn.Sequential(nn.Linear(feature_dim, _positive(hidden, "gate_hidden")),
                                       nn.SiLU(), nn.Linear(hidden, 1)))
            last = self.gate if gate_type == "linear" else self.gate[-1]
            nn.init.normal_(last.weight, std=0.02)
            nn.init.constant_(last.bias, math.log(gate_init / (1 - gate_init)))
        else:
            self.gate = None

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.length:
            raise ValueError("BoundaryTokenizer expects [B,C,configured length]")
        padded = F.pad(x, (0, self.pad)) if self.pad else x
        base = padded.unfold(-1, self.patch, self.patch)
        if self.kind == "none" or self.shift == 0:
            return base, base, base.new_zeros(base.shape[:-1])
        shifted = F.pad(padded[..., self.shift:], (0, self.shift)).unfold(-1, self.patch, self.patch)
        if self.kind == "fixed":
            gate = (0.5 * self.valid.to(x.dtype)).expand(base.shape[:-1])
        else:
            q, qs = _fp_stats(base), _fp_stats(shifted)
            mask = self.observed.to(q.dtype)
            n = mask.sum(-1)
            mu = (q * mask).sum(-1) / n
            sd = (((q - mu[..., None]).square() * mask).sum(-1) / n + self.eps).sqrt()
            vs, ms = torch.var_mean(qs, dim=-1, unbiased=False)
            ss = (vs + self.eps).sqrt()
            edge = self.patch - self.shift
            jump = qs[..., edge] - qs[..., edge - 1]
            features = [mu, sd, ms, ss, jump.abs()]
            if self.relative:
                left, right = qs[..., :edge], qs[..., edge:]
                level = right.mean(-1) - left.mean(-1)
                sl = (left[..., -1] - left[..., 0]) / max(edge - 1, 1)
                sr = (right[..., -1] - right[..., 0]) / max(self.shift - 1, 1)
                rel = torch.log1p(jump.abs() / (0.5 * (sd + ss) + self.eps))
                features += [rel, level, sr - sl]
            features = torch.stack(features, dim=-1)
            features = torch.where(self.valid[..., None], features, 0.0)
            # Respect a deliberate .double() call; under AMP the linear is autocast.
            features = features.to(next(self.gate.parameters()).dtype)
            gate = torch.sigmoid(self.gate(features).squeeze(-1)).to(base.dtype)
            gate = gate * self.valid.to(base.dtype)
        fused = base + gate[..., None] * (shifted - base)
        fused = torch.where(self.valid[..., None], fused, base)
        return base, fused, gate


class SpectralConditioner(nn.Module):
    """All rFFT bins -> compact state; no [B,C,P,F,d_s] activations or SSM.

    Bin identity is retained by flattening in a fixed order. This is a patch
    conditioner, NOT the complex-domain interpolation/reconstruction in FITS.
    """

    def __init__(self, patch, d_model, spectral_dim):
        super().__init__()
        self.encode = nn.Linear(3 * (patch // 2 + 1), spectral_dim)
        self.project = nn.Linear(spectral_dim, d_model)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, base):
        # Explicit FP32 FFT under CUDA/CPU autocast; retain FP64 for grad checks.
        with torch.autocast(device_type=base.device.type, enabled=False):
            spectrum = torch.fft.rfft(_fp_stats(base), dim=-1, norm="ortho")
            features = torch.cat((spectrum.real, spectrum.imag,
                                  torch.log1p(spectrum.abs())), dim=-1)
        features = features.to(self.encode.weight.dtype)
        return self.strength * self.project(F.silu(self.encode(features)))


class PatchStem(nn.Module):
    def __init__(self, c, patch):
        super().__init__()
        self.patch = patch
        self.tokenizer = BoundaryTokenizer(c.seq_len, patch, c.bast_gate_type,
                                          c.bast_gate_hidden, c.bast_gate_init,
                                          c.bast_use_relative_jump, c.eps)
        self.projection = nn.Linear(patch, c.d_model)
        self.spectral = SpectralConditioner(patch, c.d_model, c.spectral_dim) if c.use_spectral else None
        self.norm = nn.LayerNorm(c.d_model)

    def forward(self, x):
        base, fused, _ = self.tokenizer(x)
        z = self.projection(fused)
        if self.spectral is not None:
            z = z + self.spectral(base)
        return self.norm(z)


class TemporalBlock(nn.Module):
    """One local mixer and one global temporal residual, after scale fusion."""

    def __init__(self, c):
        super().__init__()
        d = c.d_model
        self.backend = c.temporal_backend
        self.local_norm, self.global_norm = nn.LayerNorm(d), nn.LayerNorm(d)
        self.local = nn.Conv1d(d, d, c.local_kernel, padding=c.local_kernel // 2, groups=d)
        self.local_scale = nn.Parameter(torch.tensor(0.1))
        self.global_scale = nn.Parameter(torch.tensor(0.1))
        self.drop = nn.Dropout(c.dropout)
        if self.backend == "mamba":
            try:
                from mamba_ssm import Mamba
            except (ImportError, OSError) as exc:
                raise ImportError(
                    "temporal_backend='mamba' requires a compatible CUDA mamba_ssm "
                    "and causal-conv1d installation. For the explicitly different "
                    "CPU convolution variant set temporal_backend='conv'."
                ) from exc
            self.global_mixer = Mamba(d_model=d, d_state=c.d_state, d_conv=c.d_conv,
                                      expand=c.expand, use_fast_path=True)
        elif self.backend == "conv":
            # Explicit ablation, not a Mamba emulation. Kernel 5 with dilation 2.
            self.global_mixer = nn.Sequential(
                nn.Conv1d(d, d, 5, padding=4, dilation=2, groups=d),
                nn.GELU(), nn.Conv1d(d, 2 * d, 1), nn.GLU(dim=1),
                nn.Conv1d(d, d, 1))
        else:
            raise ValueError("temporal_backend must be 'mamba' or 'conv'")

    def forward(self, z):
        z = z + self.local_scale * self.drop(self.local(self.local_norm(z).transpose(1, 2)).transpose(1, 2))
        h = self.global_norm(z)
        if self.backend == "mamba":
            if not h.is_cuda:
                raise RuntimeError("Real Mamba execution requires CUDA; no automatic conv fallback is used")
            h = self.global_mixer(h.contiguous())
        else:
            h = self.global_mixer(h.transpose(1, 2)).transpose(1, 2)
        return z + self.global_scale * self.drop(h)


class ChannelRouter(nn.Module):
    """Window-summary cross-variable routing with O(B*C*R*d_c) attention.

    Inspired by Crossformer's gather/distribute principle. Unlike its TSA,
    routing is done once per window, not separately at every patch position.
    Batch elements never communicate. There is no [C,C] attention matrix.
    """

    def __init__(self, d_model, channel_dim, routers, gated=True, gate_init=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.to_qkv = nn.Linear(d_model, 3 * channel_dim)
        self.routers = nn.Parameter(torch.randn(routers, channel_dim) * 0.02)
        self.out = nn.Linear(channel_dim, 2 * d_model)
        self.gated = gated
        # Initial effective residual strength is 0.01 in both gate settings.
        self.alpha = nn.Parameter(torch.tensor(0.01 / gate_init if gated else 0.01))
        self.scale = channel_dim ** -0.5
        if gated:
            self.token_gate = nn.Linear(d_model, 1, bias=False)
            self.context_gate = nn.Linear(d_model + channel_dim, 1)
            nn.init.zeros_(self.token_gate.weight)
            nn.init.zeros_(self.context_gate.weight)
            nn.init.constant_(self.context_gate.bias, _logit(gate_init, "channel_gate_init"))
        else:
            self.token_gate = None
            self.context_gate = None

    def forward(self, h, pool_weight, *, return_aux=False):
        summary = 0.5 * ((h * pool_weight).sum(dim=2) + h[:, :, -1])
        normalized_summary = self.norm(summary)
        q, k, v = self.to_qkv(normalized_summary).chunk(3, dim=-1)
        gather_logits = torch.einsum("rd,bcd->brc", self.routers, k) * self.scale
        gather = gather_logits.float().softmax(-1).to(v.dtype)
        slots = torch.matmul(gather, v)
        send_logits = torch.matmul(q, slots.transpose(-1, -2)) * self.scale
        send = send_logits.float().softmax(-1).to(v.dtype)
        message = torch.matmul(send, slots)
        context = self.out(message)
        gain, bias = context.chunk(2, dim=-1)
        update = torch.tanh(gain)[:, :, None] * h + bias[:, :, None]
        gate = None
        if self.gated:
            # Factorized gate: token content + variable summary/routed message.
            # Cost remains linear in C and P; there is no [C,C] attention.
            token_logits = self.token_gate(self.norm(h))  # [B,C,P,1]
            context_logits = self.context_gate(
                torch.cat((normalized_summary, message), dim=-1))[:, :, None]
            gate = torch.sigmoid(token_logits + context_logits)
            update = update * gate
        output = h + self.alpha * update
        if return_aux:
            return output, None if gate is None else gate.squeeze(-1)
        return output


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        defaults = Config(1, 1, 1)
        values = {key: _get(configs, key, getattr(defaults, key)) for key in defaults.__dataclass_fields__}
        for key in ("seq_len", "pred_len", "enc_in"):
            values[key] = _positive(_get(configs, key), key)
        for key in ("d_model", "d_state", "d_conv", "expand", "spectral_dim", "local_kernel",
                    "head_rank", "bast_gate_hidden", "channel_dim", "num_routers", "fusion_hidden"):
            values[key] = _positive(values[key], key)
        for key in ("bast_use_relative_jump", "use_spectral", "adaptive_scale", "channel_mixing", "linear_skip", "channel_gate"):
            values[key] = _boolean(values[key], key)
        patches = values["patch_sizes"]
        if isinstance(patches, str):
            patches = tuple(int(p.strip()) for p in patches.split(","))
        patches = tuple(sorted(_positive(p, "patch_sizes") for p in patches))
        if not patches or any(p % patches[0] for p in patches):
            raise ValueError("patch_sizes must be nonempty integer multiples of the smallest patch")
        if not 0 <= values["dropout"] < 1 or values["local_kernel"] % 2 == 0 or values["eps"] <= 0:
            raise ValueError("Require 0 <= dropout < 1, odd local_kernel, eps > 0")
        if values["fusion_mode"] not in ("global", "anchor_local"):
            raise ValueError("fusion_mode must be 'global' or 'anchor_local'")
        _logit(values["fusion_residual_init"], "fusion_residual_init")
        _logit(values["channel_gate_init"], "channel_gate_init")
        values["patch_sizes"] = patches
        c = self.config = Config(**values)
        self.seq_len, self.pred_len, self.enc_in = c.seq_len, c.pred_len, c.enc_in
        self.patch_sizes = patches
        self.num_tokens = math.ceil(c.seq_len / patches[0])
        self.stems = nn.ModuleList([PatchStem(c, p) for p in patches])
        self.scale_logits = nn.Parameter(torch.zeros(len(patches)))
        self.scale_score = nn.Linear(c.d_model, 1, bias=False) if c.adaptive_scale else None
        if self.scale_score is not None:
            nn.init.normal_(self.scale_score.weight, std=0.02)
        if c.fusion_mode == "anchor_local" and len(patches) > 1:
            self.local_scale_score = nn.Sequential(
                nn.Linear(2 * c.d_model, c.fusion_hidden), nn.SiLU(),
                nn.Linear(c.fusion_hidden, 1, bias=False))
            # Start close to the global prior, then learn position dependence.
            nn.init.zeros_(self.local_scale_score[-1].weight)
            self.anchor_logit = nn.Parameter(torch.tensor(
                _logit(c.fusion_residual_init, "fusion_residual_init")))
        else:
            self.local_scale_score = None
            self.register_parameter("anchor_logit", None)
        self.position = nn.Parameter(torch.randn(1, 1, self.num_tokens, c.d_model) * 0.02)
        self.node_embed = nn.Parameter(torch.randn(1, c.enc_in, 1, c.d_model) * 0.02)
        self.temporal = TemporalBlock(c)
        self.channel = (ChannelRouter(c.d_model, c.channel_dim, c.num_routers,
                                      c.channel_gate, c.channel_gate_init)
                        if c.channel_mixing and c.enc_in > 1 else None)
        self.head_norm = nn.LayerNorm(c.d_model)
        self.head = nn.Sequential(nn.Dropout(c.dropout),
                                  nn.Linear(self.num_tokens * c.d_model, c.head_rank),
                                  nn.GELU(), nn.Dropout(c.dropout),
                                  nn.Linear(c.head_rank, c.pred_len))
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)
        self.skip = nn.Linear(c.seq_len, c.pred_len, bias=False) if c.linear_skip else None
        if self.skip is not None:
            # Start the linear path at persistence; all time coefficients learn.
            nn.init.zeros_(self.skip.weight)

    def _normalize(self, x):
        stats = _fp_stats(x)
        mean = stats.mean(1, keepdim=True).detach()
        std = (stats.var(1, keepdim=True, unbiased=False) + self.config.eps).sqrt().detach()
        return ((stats - mean) / std).transpose(1, 2), mean, std

    def _encode(self, normalized, *, return_aux=False):
        batch, channels, _ = normalized.shape
        native = [stem(normalized) for stem in self.stems]
        scores = self.scale_logits.view(1, 1, -1).expand(batch, channels, -1)
        if self.scale_score is not None:
            pooled = [(z * stem.tokenizer.pool_weight).sum(2) for z, stem in zip(native, self.stems)]
            scores = scores + torch.cat([self.scale_score(z) for z in pooled], dim=-1)
        weights = scores.float().softmax(-1).to(native[0].dtype)
        aligned = []
        for z, patch in zip(native, self.patch_sizes):
            # Base-support alignment on a common fine grid, also for odd L.
            # Coarse contexts repeat over their constituent fine patches. This
            # does not claim exact timestamp alignment of BAST shifted supports.
            ratio = patch // self.patch_sizes[0]
            if ratio != 1:
                z = z.repeat_interleave(ratio, dim=2)[:, :, :self.num_tokens]
            aligned.append(z)
        local_weights = None
        strength = None
        if self.local_scale_score is not None:
            fine = aligned[0]
            local_logits = torch.cat([
                self.local_scale_score(torch.cat((fine, z - fine), dim=-1))
                for z in aligned], dim=-1)  # [B,C,P,S]
            local_weights = (scores[:, :, None].float() + local_logits.float()).softmax(-1)
            local_weights = local_weights.to(fine.dtype)
            correction = torch.zeros_like(fine)
            for i, z in enumerate(aligned[1:], start=1):
                correction = correction + local_weights[..., i, None] * (z - fine)
            strength = torch.sigmoid(self.anchor_logit).to(fine.dtype)
            # z0 + gamma * sum_s w_s (z_s-z0).  The fine contribution has
            # coefficient 1-gamma+gamma*w0. This is not a lossless encoder.
            fused = fine + strength * correction
        else:
            fused = None
            for i, z in enumerate(aligned):
                contribution = z * weights[..., i, None, None]
                fused = contribution if fused is None else fused + contribution
        dtype = fused.dtype
        fused = fused + self.position.to(dtype) + self.node_embed.to(dtype)
        h = self.temporal(fused.reshape(batch * channels, self.num_tokens, -1))
        h = h.reshape(batch, channels, self.num_tokens, self.config.d_model)
        channel_gate = None
        if self.channel is not None:
            if return_aux:
                h, channel_gate = self.channel(
                    h, self.stems[0].tokenizer.pool_weight, return_aux=True)
            else:
                h = self.channel(h, self.stems[0].tokenizer.pool_weight)
        aux = {}
        if return_aux:
            aux["scale_weights"] = weights.detach()  # native global prior [B,C,S]
            if local_weights is not None:
                aux["local_scale_weights"] = local_weights.detach()  # [B,C,P,S]
                aux["fusion_strength"] = strength.detach()  # scalar
            if channel_gate is not None:
                aux["channel_gate"] = channel_gate.detach()  # [B,C,P]
        return h, aux

    def forward(self, x, x_mark=None, y_true=None, *, return_aux=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected [B,{self.seq_len},{self.enc_in}], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Input must be floating point")
        normalized, mean, std = self._normalize(x)
        h, aux = self._encode(normalized, return_aux=return_aux)
        prediction = self.head(self.head_norm(h).flatten(start_dim=2))
        if self.skip is not None:
            anchor = normalized[..., -1:]
            prediction = prediction + anchor + self.skip(normalized - anchor)
        prediction = prediction.transpose(1, 2) * std + mean
        if return_aux:
            return prediction, aux
        return prediction

    def parameter_report(self):
        modules = {"stems": self.stems, "temporal": self.temporal, "head": self.head,
                   "channel_router": self.channel, "linear_skip": self.skip}
        report = {name: sum(p.numel() for p in module.parameters()) if module is not None else 0
                  for name, module in modules.items()}
        report["fusion_extension"] = (sum(p.numel() for p in self.local_scale_score.parameters())
                                      + self.anchor_logit.numel()
                                      if self.local_scale_score is not None else 0)
        report["total"] = sum(p.numel() for p in self.parameters())
        report["other"] = report["total"] - sum(v for k, v in report.items() if k != "total")
        report["tokens_per_temporal_pass"] = self.num_tokens
        report["mamba_calls_per_forward"] = int(self.config.temporal_backend == "mamba")
        return report


def self_test():
    """CPU functional checks; no accuracy or CUDA/Mamba benchmark is implied."""
    torch.manual_seed(2027)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    checks = 0
    try:
        cases = [
            (95, 3, (8, 16, 32)), (96, 3, (8, 16, 32)),
            (97, 3, (8, 16, 32)), (127, 3, (8, 16, 32)),
            (17, 1, (3, 6, 9)), (1, 1, (1, 2, 4)),
            (32, 3, (8, 8, 8)), (32, 3, (8,)),
        ]
        for length, channels, patches in cases:
            c = Config(length, 7, channels, d_model=16, head_rank=8,
                       fusion_hidden=4, channel_dim=4, patch_sizes=patches,
                       dropout=0.0, temporal_backend="conv")
            model = Model(c)
            x = torch.randn(2, length, channels)
            y, aux = model(x, return_aux=True)
            assert y.shape == (2, 7, channels) and torch.isfinite(y).all()
            torch.testing.assert_close(aux["scale_weights"].sum(-1),
                                       torch.ones(2, channels))
            if "local_scale_weights" in aux:
                local = aux["local_scale_weights"]
                assert local.shape == (2, channels, model.num_tokens, len(patches))
                torch.testing.assert_close(local.sum(-1), torch.ones_like(local[..., 0]))
            if channels > 1:
                gate = aux["channel_gate"]
                assert gate.shape == (2, channels, model.num_tokens)
                torch.testing.assert_close(gate, torch.full_like(gate, 0.2))
            loss = F.mse_loss(y, torch.randn_like(y))
            loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None, f"Disconnected parameter: {name}"
                assert torch.isfinite(p.grad).all(), f"Nonfinite gradient: {name}"
            # Entire centered shortcut has a null last column by construction.
            assert torch.count_nonzero(model.skip.weight.grad[:, -1]) == 0
            for stem in model.stems:
                _, mixed, gate = stem.tokenizer(model._normalize(x)[0])
                assert mixed.shape[2] == math.ceil(length / stem.patch)
                assert torch.count_nonzero(gate[..., -1]) == 0
            if len(set(patches)) < len(patches):
                assert len(model.stems) == len(patches)
                assert model.stems[0].projection.weight is not model.stems[1].projection.weight
            checks += 1

        model = Model(Config(32, 7, 3, d_model=16, head_rank=8,
                             channel_dim=4, dropout=0.0, temporal_backend="conv")).eval()
        x = torch.randn(2, 32, 3)
        # Make both proposed gates input-dependent before invariance checks.
        with torch.no_grad():
            model.local_scale_score[-1].weight.normal_(std=0.1)
            model.channel.token_gate.weight.normal_(std=0.1)
            model.channel.context_gate.weight.normal_(std=0.1)
            y, aux = model(x, return_aux=True)
            assert aux["local_scale_weights"].std(dim=2).max() > 0
            assert aux["channel_gate"].std(dim=2).max() > 0
            torch.testing.assert_close(y[:1], model(x[:1]), rtol=2e-5, atol=2e-6)
            shift = torch.tensor([2.0, -3.0, 0.5])[None, None]
            torch.testing.assert_close(model(x + shift), y + shift, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(model(x, y_true=torch.randn_like(y)), y)
            torch.testing.assert_close(model(x, x_mark=torch.randn(2, 32, 4)), y)
        checks += 1

        # Both switches can be disabled without retaining their parameters.
        baseline = Model(Config(96, 192, 21, temporal_backend="conv",
                                fusion_mode="global", channel_gate=False))
        upgraded = Model(Config(96, 192, 21, temporal_backend="conv"))
        assert baseline.parameter_report()["total"] == 223068
        assert upgraded.parameter_report()["total"] == 227486
        assert upgraded.parameter_report()["fusion_extension"] == 4129
        checks += 1
    finally:
        torch.set_num_threads(previous_threads)
    return {"status": "passed", "functional_cases": checks,
            "backend": "conv", "cuda_mamba_tested": False}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mamba", "conv"), default="mamba")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=192)
    parser.add_argument("--enc-in", type=int, default=21)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--patch-sizes", type=str, default="8,16,32")
    parser.add_argument("--fusion-mode", choices=("global", "anchor_local"), default="anchor_local")
    parser.add_argument("--no-channel-gate", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    device = torch.device(args.device or ("cuda" if args.backend == "mamba" else "cpu"))
    if args.backend == "mamba" and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("Mamba execution needs CUDA. Use --backend conv only for the distinct conv variant.")
    model = Model(Config(args.seq_len, args.pred_len, args.enc_in,
                         d_model=args.d_model, temporal_backend=args.backend,
                         patch_sizes=args.patch_sizes, fusion_mode=args.fusion_mode,
                         channel_gate=not args.no_channel_gate)).to(device)
    print(json.dumps(model.parameter_report(), indent=2))
    if args.smoke_test:
        x = torch.randn(2, args.seq_len, args.enc_in, device=device)
        target = torch.randn(2, args.pred_len, args.enc_in, device=device)
        prediction, aux = model(x, return_aux=True)
        loss = F.mse_loss(prediction, target)
        loss.backward()
        assert torch.isfinite(loss)
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        print(json.dumps({"shape": list(prediction.shape), "loss": loss.item(),
                          "aux_shapes": {k: list(v.shape) for k, v in aux.items()},
                          "forward_backward": "passed"}, indent=2))


if __name__ == "__main__":
    main()
