"""Paired fine-tune objectives and common validation metrics.

This module does not train or evaluate a model when imported. ``loss_terms``
receives fields already normalized by the original, fixed training-set RMS.
``summarize_validation`` receives physical fields and uses those same saved
output scales only to reconstruct the two normalized losses. No scales are fit
here. Quadrature weights already include dx; do not multiply by dx again.

The amplitude floor is used only in the mixed objective. Reported relative L2
errors always divide by the actual physical target norm, without a floor.
"""

from __future__ import annotations

import numpy as np
import torch


FIELD_NAMES = ("E", "H", "P")


def _real_array(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _controls(mix_lambda, delta):
    coefficient = _real_array(mix_lambda, "mix_lambda")
    if coefficient.ndim != 0 or not 0.0 <= float(coefficient) <= 1.0:
        raise ValueError("mix_lambda must be a finite scalar in [0, 1]")
    floor = _real_array(delta, "delta")
    if floor.shape != (3,) or np.any(floor <= 0):
        raise ValueError("delta must contain three positive normalized amplitude floors")
    return float(coefficient), floor


def loss_terms(pred, target, weights, mix_lambda, delta):
    """Compute original MSE and mixed loss without detaching predictions.

    Parameters
    ----------
    pred, target : torch.Tensor, [B, 3, nx]
        E/H/P values in the existing normalized field coordinates.
    weights : array-like, [nx]
        Fixed spatial integration weights, including the spatial measure.
    mix_lambda : float
        Relative-term mixing coefficient in [0, 1].
    delta : array-like, [3]
        Positive normalized amplitude floors for E, H, and P respectively.

    Returns a dictionary containing differentiable scalar ``mse`` and
    ``mixed`` losses and the per-sample/field tensors ``D`` and ``A2``.
    D_ic = sum_j(w_j * (pred-target)^2) / sum_j(w_j).
    A2_ic = sum_j(w_j * target^2) / sum_j(w_j).
    mixed = mean_ic((1-lambda)*D_ic + lambda*D_ic/(A2_ic+delta_c^2)).
    """
    coefficient, floor = _controls(mix_lambda, delta)
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("pred and target must be torch tensors")
    if pred.ndim != 3 or pred.shape != target.shape or pred.shape[1] != 3:
        raise ValueError("pred and target must have identical shape [B, 3, nx]")
    if pred.shape[0] == 0 or pred.shape[-1] < 2:
        raise ValueError("loss_terms requires a nonempty batch and nx >= 2")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError("pred and target must use real floating-point dtypes")
    if pred.device != target.device or pred.dtype != target.dtype:
        raise ValueError("pred and target must have the same device and dtype")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("pred and target must be finite")
    if isinstance(weights, torch.Tensor) and weights.is_complex():
        raise ValueError("weights must be real-valued")
    if not isinstance(weights, torch.Tensor) and np.iscomplexobj(weights):
        raise ValueError("weights must be real-valued")
    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    if w.ndim != 1 or w.numel() != pred.shape[-1]:
        raise ValueError("weights must have shape [nx]")
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("weights must be finite and nonnegative")
    weight_sum = w.sum()
    if not bool(torch.isfinite(weight_sum)) or not bool(weight_sum > 0):
        raise ValueError("weights must have a finite positive sum")
    floor_tensor = torch.as_tensor(floor, dtype=pred.dtype, device=pred.device)
    floor_squared = floor_tensor.square()
    if not bool(torch.isfinite(floor_squared).all()) or not bool((floor_squared > 0).all()):
        raise ValueError("squared delta must remain finite and positive in the field dtype")
    D = ((pred - target).square() * w).sum(dim=-1) / weight_sum
    A2 = (target.square() * w).sum(dim=-1) / weight_sum
    mixed_terms = (1.0 - coefficient) * D + coefficient * D / (A2 + floor_squared)
    if not bool(torch.isfinite(D).all()) or not bool(torch.isfinite(A2).all()):
        raise ValueError("weighted normalized field squares overflowed")
    if not bool(torch.isfinite(mixed_terms).all()):
        raise ValueError("mixed loss is nonfinite")
    return {"mse": D.mean(), "mixed": mixed_terms.mean(), "D": D, "A2": A2}


def _summary(values):
    array = np.asarray(values, dtype=np.float64).ravel()
    valid = array[np.isfinite(array)]
    result = {
        "mean": None,
        "median": None,
        "p95": None,
        "max": None,
        "defined_count": int(valid.size),
        "undefined_count": int(array.size - valid.size),
    }
    if valid.size:
        result.update(
            mean=float(np.mean(valid)),
            median=float(np.median(valid)),
            p95=float(np.quantile(valid, 0.95)),
            max=float(np.max(valid)),
        )
    return result


def summarize_validation(
    prediction_physical,
    target_physical,
    weights,
    output_scale,
    mix_lambda,
    delta,
    relative_norm_threshold=1e-10,
):
    """Summarize a complete validation set using common physical metrics.

    Undefined near-zero-denominator relative errors are excluded from each
    field's relative summary, counted explicitly, and never treated as passes.
    If any field has no defined relative errors, the common macro mean is None
    so the caller can stop rather than silently change the comparison metric.
    Every returned scalar is JSON-safe; no predictions or reference fields are
    altered. This function only computes on arrays already supplied by callers.
    """
    coefficient, floor = _controls(mix_lambda, delta)
    pred = _real_array(prediction_physical, "prediction_physical")
    target = _real_array(target_physical, "target_physical")
    w = _real_array(weights, "weights")
    scales = _real_array(output_scale, "output_scale")
    threshold_array = _real_array(relative_norm_threshold, "relative_norm_threshold")
    if threshold_array.ndim != 0 or float(threshold_array) < 0:
        raise ValueError("relative_norm_threshold must be a finite nonnegative scalar")
    threshold = float(threshold_array)
    if pred.ndim != 3 or pred.shape != target.shape or pred.shape[1] != 3:
        raise ValueError("physical predictions and targets must have shape [B, 3, nx]")
    if pred.shape[0] == 0 or pred.shape[-1] < 2:
        raise ValueError("summarize_validation requires nonempty samples and nx >= 2")
    if w.shape != (pred.shape[-1],) or np.any(w < 0):
        raise ValueError("weights must be nonnegative and have shape [nx]")
    weight_sum = float(np.sum(w))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("weights must have a finite positive sum")
    if scales.shape != (3,) or np.any(scales <= 0):
        raise ValueError("output_scale must contain the three positive saved field scales")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        absolute_squared = np.sum((pred - target) ** 2 * w, axis=-1)
        reference_squared = np.sum(target**2 * w, axis=-1)
        absolute = np.sqrt(absolute_squared)
        reference = np.sqrt(reference_squared)
        D = absolute_squared / (weight_sum * scales[None, :] ** 2)
        A2 = reference_squared / (weight_sum * scales[None, :] ** 2)
        floor_squared = floor**2
        mixed_terms = (1.0 - coefficient) * D + coefficient * D / (A2 + floor_squared)
    if not np.isfinite(floor_squared).all() or np.any(floor_squared <= 0):
        raise ValueError("squared delta must remain finite and positive")
    if any(not np.isfinite(a).all() for a in (absolute, reference, D, A2, mixed_terms)):
        raise ValueError("physical metrics or normalized losses overflowed")
    defined = reference > threshold
    relative = np.full_like(reference, np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(absolute, reference, out=relative, where=defined)
    if not np.isfinite(relative[defined]).all():
        raise ValueError("a defined relative L2 error is nonfinite")
    fields = {
        name: {
            "absolute_l2": _summary(absolute[:, c]),
            "relative_l2": _summary(relative[:, c]),
            "reference_l2": _summary(reference[:, c]),
        }
        for c, name in enumerate(FIELD_NAMES)
    }
    relative_means = [fields[name]["relative_l2"]["mean"] for name in FIELD_NAMES]
    macro_relative = (
        None if any(value is None for value in relative_means)
        else float(np.mean(relative_means))
    )
    joint_defined = np.all(defined, axis=1)
    passed = joint_defined & np.all(relative <= 0.01, axis=1)
    return {
        "sample_count": int(pred.shape[0]),
        "mse": float(np.mean(D)),
        "mixed_loss": float(np.mean(mixed_terms)),
        "mix_lambda": coefficient,
        "delta_normalized": floor.tolist(),
        "macro_mean_relative_l2": macro_relative,
        "fields": fields,
        "joint_relative_le_1pct": {
            "pass_count": int(np.count_nonzero(passed)),
            "total_count": int(pred.shape[0]),
            "undefined_count": int(np.count_nonzero(~joint_defined)),
            "fraction_all": float(np.count_nonzero(passed) / pred.shape[0]),
        },
        "relative_norm_threshold": threshold,
        "relative_error_definition": "physical absolute L2 / physical target L2; no delta",
        "relative_error_units": "fraction; multiply by 100 for percent",
        "common_metric_definition": "equal mean of E/H/P per-field mean true relative L2",
        "quadrature": "supplied spatial weights already include dx",
        "metric_scope": "weighted discrete field errors on the saved sample grid",
    }
