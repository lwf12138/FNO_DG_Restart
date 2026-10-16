"""Training loss and physical-space metrics; importing this module runs no model.

The supplied quadrature weights already include the spatial measure. Do not
multiply by dx again. Undefined relative errors remain NaN in arrays, with an
explicit mask; their JSON summaries use None and their CSV entries are empty.
"""

from __future__ import annotations

try:
    import numpy as np
    import torch
except ImportError as exc:
    raise ImportError(
        "FNO metrics require NumPy and PyTorch. Install the documented training "
        "dependencies explicitly before running these scripts."
    ) from exc


FIELD_NAMES = ("E", "H", "P")


def weighted_mse(pred, target, weights):
    """Equal-field normalized MSE: sum(w * error**2)/(B * 3 * sum(w))."""
    if pred.ndim != 3 or pred.shape != target.shape or pred.shape[1] != 3:
        raise ValueError("pred and target must have identical shape [B, 3, nx]")
    if pred.shape[0] == 0 or pred.shape[-1] < 2:
        raise ValueError("weighted_mse requires a nonempty batch and at least 2 points")
    if pred.is_complex() or target.is_complex():
        raise ValueError("E/H/P predictions and targets must be real-valued")
    if pred.device != target.device:
        raise ValueError("pred and target must be on the same device")
    w = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    if w.ndim != 1 or w.numel() != pred.shape[-1]:
        raise ValueError("weights must have shape [nx]")
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
        raise ValueError("quadrature weights must be finite and nonnegative")
    weight_sum = w.sum()
    if not bool(weight_sum > 0):
        raise ValueError("quadrature weights must have a positive sum")
    return ((pred - target).square() * w).sum() / (pred.shape[0] * 3 * weight_sum)


def _summary(values):
    """JSON-safe descriptive statistics, excluding undefined entries."""
    values = np.asarray(values, dtype=np.float64).ravel()
    valid = values[np.isfinite(values)]
    result = {
        "total_count": int(values.size),
        "valid_count": int(valid.size),
        "undefined_count": int(values.size - valid.size),
        "mean": None,
        "median": None,
        "p95": None,
        "max": None,
    }
    if valid.size:
        result.update(
            mean=float(np.mean(valid)),
            median=float(np.median(valid)),
            p95=float(np.quantile(valid, 0.95)),
            max=float(np.max(valid)),
        )
    return result


def physical_metrics(pred_np, true_np, weights, relative_norm_threshold=1e-10):
    """Return per-sample/field L2 errors and explicitly separated boundary metrics.

    Inputs have shape [B, 3, nx] in physical units. ``boundary_error`` is the
    signed difference from the DG target at the two endpoints, [B, 3, 2].
    ``e_zero_boundary_residual`` is the predicted electric field at the two
    endpoints, [B, 2], relative to the physical zero boundary condition.
    These are different quantities because DG endpoint traces need not vanish.
    Aggregate boundary statistics describe absolute magnitudes.
    """
    pred = np.asarray(pred_np, dtype=np.float64)
    true = np.asarray(true_np, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    threshold = float(relative_norm_threshold)
    if pred.ndim != 3 or pred.shape != true.shape or pred.shape[1] != 3:
        raise ValueError("physical predictions/targets must have shape [B, 3, nx]")
    if pred.shape[0] == 0 or pred.shape[-1] < 2:
        raise ValueError("physical_metrics requires nonempty samples and nx >= 2")
    if w.ndim != 1 or w.size != pred.shape[-1]:
        raise ValueError("quadrature weights must have shape [nx]")
    if not np.isfinite(w).all() or np.any(w < 0) or not float(w.sum()) > 0:
        raise ValueError("quadrature weights must be finite, nonnegative, and nonzero")
    if not np.isfinite(pred).all() or not np.isfinite(true).all():
        raise ValueError("physical predictions and targets must be finite")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("relative_norm_threshold must be finite and nonnegative")

    difference = pred - true
    absolute = np.sqrt(np.sum(difference**2 * w, axis=-1))
    reference_norm = np.sqrt(np.sum(true**2 * w, axis=-1))
    if not np.isfinite(absolute).all() or not np.isfinite(reference_norm).all():
        raise ValueError("physical L2 integration overflowed")
    defined = reference_norm > threshold
    relative = np.full_like(absolute, np.nan)
    np.divide(absolute, reference_norm, out=relative, where=defined)
    if not np.isfinite(relative[defined]).all():
        raise ValueError("a defined relative L2 error is nonfinite")
    boundary_error = difference[..., [0, -1]]
    e_zero_boundary_residual = pred[:, 0, :][:, [0, -1]]

    per_field = {}
    for index, name in enumerate(FIELD_NAMES):
        per_field[name] = {
            "absolute_l2": _summary(absolute[:, index]),
            "relative_l2": _summary(relative[:, index]),
            "reference_l2": _summary(reference_norm[:, index]),
            "absolute_boundary_error_against_dg": {
                "left": _summary(np.abs(boundary_error[:, index, 0])),
                "right": _summary(np.abs(boundary_error[:, index, 1])),
            },
        }
    aggregate = {
        "relative_norm_threshold": threshold,
        "relative_error_definition": "absolute L2 / reference L2 when reference L2 > threshold",
        "relative_error_units": "fraction (multiply by 100 for percent)",
        "quadrature": "supplied physical weights; dx is not multiplied a second time",
        "metric_scope": f"weighted discrete field errors on the current {pred.shape[-1]}-point grid; not cellwise Gauss, flux, or superconvergence errors",
        "per_field": per_field,
        "absolute_e_zero_boundary_residual": {
            "left": _summary(np.abs(e_zero_boundary_residual[:, 0])),
            "right": _summary(np.abs(e_zero_boundary_residual[:, 1])),
        },
    }
    return {
        "absolute_l2": absolute,
        "relative_l2": relative,
        "reference_l2": reference_norm,
        "relative_defined": defined,
        "boundary_error": boundary_error,
        "e_zero_boundary_residual": e_zero_boundary_residual,
        "aggregate": aggregate,
    }
