"""Configuration, provenance and explicit device selection; no work on import."""
from collections import OrderedDict
import copy
from datetime import datetime
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import uuid

ROOT = Path(__file__).resolve().parent


def add_config_arguments(parser):
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                        help='Override e.g. training.epochs=100 or training.device="cpu"')


def resolve_path(value, cfg):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(cfg["_config_path"]).parent / path).resolve()


def load_config(path=None, overrides=()):
    path = Path(path or ROOT / "config.json").expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = json.load(handle)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE: {item}")
        key, text = item.split("=", 1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown configuration section: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {key}")
        target[parts[-1]] = value
    cfg["_config_path"] = str(path)
    for key in ("raw_path", "raw_config_path", "prepared_path", "normalizer_path"):
        cfg["data"][key] = str(resolve_path(cfg["data"][key], cfg))
    cfg["paths"]["results_dir"] = str(resolve_path(cfg["paths"]["results_dir"], cfg))
    for key in ("checkpoint_path", "prediction_input_path"):
        if cfg.get(key) is not None:
            cfg[key] = str(resolve_path(cfg[key], cfg))
    if cfg["data"]["target_time"] != 1.0:
        raise ValueError("This workflow is fixed to the uniquely stored T=1 frame.")
    return cfg


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def new_run_dir(base, prefix="train"):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    path = base / f"{prefix}_{name}"
    path.mkdir(exist_ok=False)
    return path


def environment_info(device=None):
    versions = {}
    for package in ("torch", "neuraloperator", "numpy", "scipy", "tensorly", "tensorly-torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "NOT INSTALLED"
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "versions": versions, "selected_device": None if device is None else str(device),
            "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")}


def select_device(name="cpu"):
    import torch
    if not isinstance(name, str):
        raise ValueError("device must be cpu, mps, cuda, or cuda:<index>")
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("MPS CPU fallback is enabled. Unset PYTORCH_ENABLE_MPS_FALLBACK; "
                               "this workflow must not silently mix CPU and MPS. Or select cpu.")
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("Explicit MPS request unavailable. Select device='cpu' explicitly.")
        return torch.device("mps")
    if name == "cuda" or (name.startswith("cuda:") and name[5:].isdigit()):
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA request unavailable. Select device='cpu' explicitly.")
        index = torch.cuda.current_device() if name == "cuda" else int(name[5:])
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist; no device fallback is performed.")
        return torch.device("cuda", index)
    raise ValueError(f"Unsupported explicit device {name!r}; use cpu, mps or cuda[:index].")


def seed_all(seed, device):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def capture_rng(generator, device):
    import numpy as np
    import torch
    state = np.random.get_state()
    result = {"python": random.getstate(),
              "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
              "torch_cpu": torch.get_rng_state(), "loader": generator.get_state(),
              "device_type": device.type}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state, generator, device):
    import numpy as np
    import torch
    if state["device_type"] != device.type:
        raise ValueError("Exact-state resume requires the same device type; start a new run otherwise.")
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0], np.array(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["loader"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["mps"])


def cpu_model_state(model):
    import torch
    source = model.state_dict()
    state = OrderedDict()
    for key, value in source.items():
        # neuraloperator 2.0 adds constructor kwargs as a top-level dictionary.
        # It is not a weight/buffer (and can contain Python callables). Our
        # checkpoint already stores the validated constructor in model_config.
        if key == "_metadata" and isinstance(value, dict):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unexpected non-tensor model state at {key!r}: {type(value).__name__}")
        state[key] = value.detach().cpu().clone()
    # Preserve PyTorch's separate module-version metadata attribute.
    if hasattr(source, "_metadata"):
        state._metadata = copy.deepcopy(source._metadata)
    return state


def save_checkpoint(path, payload):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    """Load only your own trusted checkpoint; no arbitrary whole-model unpickling."""
    import torch
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "model_state_dict", "model_config", "normalizer", "protocol",
                "data_fingerprints", "epoch", "best_validation_loss", "training_config", "environment"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Not a checkpoint created by this FNO workflow.")
    if payload["format_version"] != 1:
        raise ValueError("Unsupported checkpoint format version.")
    if not math.isfinite(float(payload["best_validation_loss"])):
        raise ValueError("Checkpoint has no finite validation metric.")
    return payload
