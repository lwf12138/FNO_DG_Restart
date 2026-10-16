"""Thin wrapper of official neuraloperator 2.0.0; no custom Fourier convolution."""
import copy
from importlib import metadata
import math
import torch
from torch import nn
import torch.nn.functional as F

MODEL_KEYS = {
    "in_channels", "out_channels", "hidden_channels", "n_layers", "n_modes",
    "positional_embedding", "factorization", "norm", "activation", "precision",
    "domain_padding", "use_channel_mlp", "channel_mlp_expansion", "channel_mlp_dropout",
    "channel_mlp_skip", "fno_skip", "lifting_channel_ratio", "projection_channel_ratio",
    "implementation", "fno_block_precision",
}


def validate_model_config(config):
    if set(config) != MODEL_KEYS:
        raise ValueError(f"Model keys differ: unknown={set(config)-MODEL_KEYS}, missing={MODEL_KEYS-set(config)}")
    fixed = dict(in_channels=5, out_channels=3, positional_embedding=None, factorization=None,
                 norm=None, activation="gelu", precision="float32", implementation="reconstructed",
                 fno_block_precision="full", fno_skip="linear", channel_mlp_skip="soft-gating",
                 use_channel_mlp=True, channel_mlp_dropout=0.0)
    for key, value in fixed.items():
        if config[key] != value:
            raise ValueError(f"This standard FNO workflow requires {key}={value!r}.")
    for key in ("hidden_channels", "n_layers"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    modes = config["n_modes"]
    if not isinstance(modes, (list, tuple)) or len(modes) != 1 or type(modes[0]) is not int:
        raise ValueError("n_modes must be a one-element integer list/tuple for one spatial dimension.")
    if modes[0] < 2 or modes[0] % 2:
        raise ValueError("Use a positive even n_modes API value; 16 retains 9 real-FFT coefficients.")
    padding = config["domain_padding"]
    if not isinstance(padding, (int, float)) or not math.isfinite(padding) or not 0 <= padding <= 1:
        raise ValueError("domain_padding must be a finite per-side fraction between 0 and 1.")
    for key in ("lifting_channel_ratio", "projection_channel_ratio", "channel_mlp_expansion"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if round(config["hidden_channels"] * config["channel_mlp_expansion"]) < 1:
        raise ValueError("channel_mlp_expansion produces an empty hidden layer.")


class StandardFNO1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_model_config(config)
        try:
            installed = metadata.version("neuraloperator")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError("neuraloperator is missing. Manually install this directory's requirements.txt "
                               "into your selected PyCharm interpreter.") from exc
        if installed != "2.0.0":
            raise RuntimeError(f"Expected neuraloperator==2.0.0, found {installed}; API versions are not mixed.")
        if torch.__version__.split("+")[0] != "2.8.0":
            raise RuntimeError(f"This implementation targets torch==2.8.0; found {torch.__version__}.")
        if torch.get_default_dtype() != torch.float32:
            raise RuntimeError("FNO construction requires the standard float32 default dtype.")
        from neuralop.models import FNO
        self.config = copy.deepcopy(config)
        kwargs = {key: value for key, value in config.items() if key not in {"activation", "precision"}}
        kwargs["n_modes"] = tuple(config["n_modes"])
        kwargs["non_linearity"] = F.gelu
        self.operator = FNO(**kwargs)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 5 or x.shape[0] < 1 or x.shape[-1] < 3:
            raise ValueError("Expected nonempty X[batch,5,nx] with nx>=3.")
        if x.dtype != torch.float32:
            raise ValueError(f"Expected float32 physical-input channels, got {x.dtype}.")
        if x.device != next(self.parameters()).device:
            raise RuntimeError("Input and model must be on the same explicitly selected device.")
        nx = x.shape[-1]
        padded_nx = nx + 2 * round(nx * self.config["domain_padding"])
        if self.config["n_modes"][0] // 2 + 1 > padded_nx // 2 + 1:
            raise ValueError("Requested spectral modes exceed this input's padded FFT size.")
        y = self.operator(x)
        if y.shape != (x.shape[0], 3, nx):
            raise RuntimeError(f"Official FNO returned {tuple(y.shape)} instead of [batch,3,{nx}].")
        return y


def build_model(model_config):
    return StandardFNO1d(model_config)
