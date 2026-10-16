"""Predict physical terminal E/H/P from a self-contained checkpoint and initial data.

This entry point reads no training dataset, DG solver, analytic reference,
initial-mode coefficients, or future field values. Grid changes are rejected.
Importing the module defines functions only and does not load a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

try:
    import numpy as np
    import torch
except ImportError as exc:
    raise ImportError(
        "Prediction requires NumPy, PyTorch, and the documented neuraloperator "
        "dependencies. No dependency is installed automatically."
    ) from exc

if __package__:
    from .data import decode_outputs, encode_inputs, protocol_from_normalizer, validate_normalizer
    from .model import build_model
    from .utils import (
        add_config_arguments, load_checkpoint, load_config, new_run_dir,
        resolve_path, select_device, write_json,
    )
else:
    from data import decode_outputs, encode_inputs, protocol_from_normalizer, validate_normalizer
    from model import build_model
    from utils import (
        add_config_arguments, load_checkpoint, load_config, new_run_dir,
        resolve_path, select_device, write_json,
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _predict_single(checkpoint_path, alpha, tau, E0, H0, x, device="cpu"):
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    normalizer = checkpoint["normalizer"]
    validate_normalizer(normalizer)
    if json.dumps(checkpoint["protocol"], sort_keys=True, allow_nan=False) != json.dumps(
        protocol_from_normalizer(normalizer), sort_keys=True, allow_nan=False
    ):
        raise ValueError("Checkpoint protocol does not agree with its stored normalizer")
    # Preserve input dtype until encode_inputs validates real, finite fields.
    grid = np.asarray(x)
    electric_initial = np.asarray(E0)
    magnetic_initial = np.asarray(H0)
    if grid.ndim != 1 or electric_initial.shape != grid.shape or magnetic_initial.shape != grid.shape:
        raise ValueError("Single-sample x, E0, and H0 must be one-dimensional arrays with the same shape")
    alpha_array, tau_array = np.asarray(alpha), np.asarray(tau)
    if alpha_array.size != 1 or tau_array.size != 1:
        raise ValueError("predict_fields accepts one scalar alpha and tau")
    alpha_value = float(alpha_array.reshape(()))
    tau_value = float(tau_array.reshape(()))
    # encode_inputs validates physical ranges, finite values, and the entire
    # checkpoint grid. It does not clip parameters or resample initial fields.
    features = encode_inputs(
        alpha_value, tau_value, electric_initial, magnetic_initial, grid, normalizer,
    )
    if features.shape != (1, 5, grid.size):
        raise ValueError(f"Input encoder returned unexpected shape {features.shape}")
    selected_device = select_device(device)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(selected_device)
    model.eval()
    with torch.inference_mode():
        inputs = torch.from_numpy(np.ascontiguousarray(features)).to(selected_device)
        output = model(inputs).detach().cpu().numpy()
    if output.shape != (1, 3, grid.size):
        raise ValueError(f"Model returned unexpected prediction shape {output.shape}")
    physical = np.asarray(decode_outputs(output, normalizer)[0], dtype=np.float64)
    if not np.isfinite(physical).all():
        raise ValueError("The model produced nonfinite physical fields")
    metadata = {
        "operation": "single_sample_prediction", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]), "alpha": alpha_value, "tau": tau_value,
        "field_order": ["E", "H", "P"], "output_shape": list(physical.shape),
        "model_config": checkpoint["model_config"], "protocol": checkpoint["protocol"],
        "grid_policy": "full checkpoint grid validated; no resampling or clipping",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": str(np.__version__), "torch": str(torch.__version__),
            "device": str(selected_device),
        },
    }
    return physical, metadata


def predict_fields(checkpoint_path, alpha, tau, E0, H0, x, device="cpu"):
    """Return physical E/H/P as a NumPy array [3, nx] for one initial condition.

    The checkpoint provides normalization and the model definition. Inputs must
    use its full grid and declared material range. This function writes no files.
    Relative checkpoint paths are based on this script's directory, not cwd.
    """
    fields, _ = _predict_single(checkpoint_path, alpha, tau, E0, H0, x, device)
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", help="Explicit checkpoint path; otherwise use config.checkpoint_path")
    parser.add_argument("--input", help="NPZ containing only alpha, tau, E0, H0, x")
    parser.add_argument("--device", help="Override configured device; no silent fallback")
    parser.add_argument("--output", help="New output directory; an existing path is always rejected")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    checkpoint_value = args.checkpoint or cfg.get("checkpoint_path")
    input_value = args.input or cfg.get("prediction_input_path")
    if not checkpoint_value:
        parser.error("Specify --checkpoint or config.checkpoint_path; latest is never selected automatically")
    if not input_value:
        parser.error("Specify --input or config.prediction_input_path")
    checkpoint_path = Path(resolve_path(checkpoint_value, cfg))
    input_path = Path(resolve_path(input_value, cfg))
    explicit_output = Path(resolve_path(args.output, cfg)) if args.output else None
    if explicit_output is not None and explicit_output.exists():
        raise FileExistsError(f"Refusing to overwrite prediction output: {explicit_output}")
    required = {"alpha", "tau", "E0", "H0", "x"}
    with np.load(input_path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError(f"Prediction NPZ must contain exactly {sorted(required)}; found {sorted(data.files)}")
        inputs = {key: data[key].copy() for key in required}
    fields, metadata = _predict_single(
        checkpoint_path, inputs["alpha"], inputs["tau"], inputs["E0"], inputs["H0"], inputs["x"],
        args.device or cfg["training"]["device"],
    )
    if explicit_output is None:
        output_dir = new_run_dir(resolve_path(cfg["paths"]["results_dir"], cfg), "prediction")
    else:
        # exist_ok=False also rejects a concurrent writer creating this path.
        explicit_output.mkdir(parents=True, exist_ok=False)
        output_dir = explicit_output
    metadata["input_path"] = str(input_path.resolve())
    metadata["input_sha256"] = _sha256(input_path)
    np.savez_compressed(
        output_dir / "prediction.npz", fields=fields, E=fields[0], H=fields[1], P=fields[2],
        x=np.asarray(inputs["x"], dtype=np.float64), alpha=metadata["alpha"], tau=metadata["tau"],
        E0=np.asarray(inputs["E0"], dtype=np.float64), H0=np.asarray(inputs["H0"], dtype=np.float64),
    )
    write_json(output_dir / "summary.json", metadata)
    print(f"Prediction saved to {output_dir}")


if __name__ == "__main__":
    main()
