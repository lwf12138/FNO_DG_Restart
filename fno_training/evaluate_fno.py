"""Explicit evaluation of one checkpoint; this module never trains a model.

Run this entry point yourself when evaluation is desired. It does not discover
the latest checkpoint and it does not substitute a model from the current
configuration for the architecture recorded in the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

try:
    import numpy as np
    import torch
except ImportError as exc:
    raise ImportError(
        "Evaluation requires NumPy, PyTorch, and the documented neuraloperator "
        "dependencies. No dependency is installed automatically."
    ) from exc

if __package__:
    from .data import decode_outputs, load_prepared, protocol_from_normalizer, validate_normalizer
    from .metrics import FIELD_NAMES, physical_metrics
    from .model import build_model
    from .utils import (
        add_config_arguments, load_checkpoint, load_config, new_run_dir,
        resolve_path, select_device, write_json,
    )
else:
    from data import decode_outputs, load_prepared, protocol_from_normalizer, validate_normalizer
    from metrics import FIELD_NAMES, physical_metrics
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


def _require_same(actual, expected, label):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError(f"Checkpoint {label} does not match the prepared data")


def _write_per_sample(path, prepared, indices, split_name, metrics):
    columns = [
        "sample_id", "material_id", "split", "field", "absolute_l2",
        "reference_l2", "relative_l2", "relative_defined",
        "boundary_error_left_against_dg", "boundary_error_right_against_dg",
        "e_zero_boundary_residual_left", "e_zero_boundary_residual_right",
    ]
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row_index, data_index in enumerate(indices):
            for field_index, name in enumerate(FIELD_NAMES):
                defined = bool(metrics["relative_defined"][row_index, field_index])
                row = {
                    "sample_id": str(prepared.sample_id[data_index]),
                    "material_id": str(prepared.material_id[data_index]),
                    "split": split_name,
                    "field": name,
                    "absolute_l2": float(metrics["absolute_l2"][row_index, field_index]),
                    "reference_l2": float(metrics["reference_l2"][row_index, field_index]),
                    "relative_l2": float(metrics["relative_l2"][row_index, field_index]) if defined else "",
                    "relative_defined": int(defined),
                    "boundary_error_left_against_dg": float(metrics["boundary_error"][row_index, field_index, 0]),
                    "boundary_error_right_against_dg": float(metrics["boundary_error"][row_index, field_index, 1]),
                    "e_zero_boundary_residual_left": "",
                    "e_zero_boundary_residual_right": "",
                }
                if field_index == 0:
                    row["e_zero_boundary_residual_left"] = float(metrics["e_zero_boundary_residual"][row_index, 0])
                    row["e_zero_boundary_residual_right"] = float(metrics["e_zero_boundary_residual"][row_index, 1])
                writer.writerow(row)


def evaluate_checkpoint(cfg, checkpoint_path, split_name="test", device_name=None, batch_size=None):
    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    checkpoint_path = resolve_path(checkpoint_path, cfg)
    checkpoint = load_checkpoint(checkpoint_path)
    validate_normalizer(checkpoint["normalizer"])
    _require_same(
        checkpoint["protocol"], protocol_from_normalizer(checkpoint["normalizer"]),
        "protocol derived from its normalizer",
    )
    prepared = load_prepared(cfg)
    _require_same(checkpoint["normalizer"], prepared.normalizer, "normalizer")
    _require_same(checkpoint["protocol"], prepared.protocol, "protocol")
    _require_same(checkpoint["data_fingerprints"], prepared.fingerprints, "data fingerprints")
    indices = np.asarray(prepared.indices(split_name), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"The {split_name} split has no samples")
    selected_batch_size = cfg["evaluation"]["batch_size"] if batch_size is None else batch_size
    if isinstance(selected_batch_size, bool) or int(selected_batch_size) != selected_batch_size or selected_batch_size < 1:
        raise ValueError("evaluation batch_size must be a positive integer")
    selected_batch_size = int(selected_batch_size)
    selected_device = select_device(device_name or cfg["training"]["device"])
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(selected_device)
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(indices), selected_batch_size):
            selected = indices[start:start + selected_batch_size]
            # X is already normalized by preparation. Never normalize it again.
            inputs = torch.from_numpy(np.ascontiguousarray(prepared.X[selected])).to(selected_device)
            outputs = model(inputs)
            parts.append(outputs.detach().cpu().numpy())
    predicted_normalized = np.concatenate(parts, axis=0)
    expected_shape = (len(indices), 3, len(prepared.x))
    if predicted_normalized.shape != expected_shape:
        raise ValueError(f"Unexpected prediction shape {predicted_normalized.shape}; expected {expected_shape}")
    predicted_physical = decode_outputs(predicted_normalized, checkpoint["normalizer"])
    target_physical = np.asarray(prepared.Y_physical[indices], dtype=np.float64)
    metrics = physical_metrics(
        predicted_physical, target_physical, prepared.weights,
        cfg["evaluation"]["relative_norm_threshold"],
    )
    output_dir = new_run_dir(resolve_path(cfg["paths"]["results_dir"], cfg), "evaluation")
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": str(np.__version__), "torch": str(torch.__version__),
        "device": str(selected_device),
    }
    summary = {
        "operation": "evaluation", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "split": split_name, "sample_count": int(len(indices)),
        "batch_size": selected_batch_size, "field_order": list(FIELD_NAMES),
        "model_config": checkpoint["model_config"], "protocol": checkpoint["protocol"],
        "data_fingerprints": checkpoint["data_fingerprints"], "environment": environment,
        "metrics": metrics["aggregate"],
        "boundary_csv_convention": "signed prediction-DG error; signed E prediction for physical zero boundary residual",
        "undefined_relative_csv_convention": "empty relative_l2 and relative_defined=0",
        "interpretation": "This split was explicitly selected for evaluation; no model selection or training occurs here.",
    }
    np.savez_compressed(
        output_dir / "predictions.npz", prediction_physical=predicted_physical,
        target_physical=target_physical, x=prepared.x, weights=prepared.weights,
        sample_id=prepared.sample_id[indices], material_id=prepared.material_id[indices],
        split=prepared.split[indices], absolute_l2=metrics["absolute_l2"],
        reference_l2=metrics["reference_l2"], relative_l2=metrics["relative_l2"],
        relative_defined=metrics["relative_defined"], boundary_error=metrics["boundary_error"],
        e_zero_boundary_residual=metrics["e_zero_boundary_residual"],
    )
    _write_per_sample(output_dir / "per_sample.csv", prepared, indices, split_name, metrics)
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "summary.json", summary)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--checkpoint", help="Explicit checkpoint path; otherwise use config.checkpoint_path")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device", help="Override configured device; no silent fallback")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    checkpoint_value = args.checkpoint or cfg.get("checkpoint_path")
    if not checkpoint_value:
        parser.error("Specify --checkpoint or checkpoint_path in the configuration; latest is never selected automatically")
    output_dir = evaluate_checkpoint(
        cfg, resolve_path(checkpoint_value, cfg), args.split, args.device, args.batch_size,
    )
    print(f"Evaluation saved to {output_dir}")


if __name__ == "__main__":
    main()
