"""Strict NumPy data preparation/encoding; no model or device operations.

Material parameters and coordinates map to [-1, 1]. Fields are divided by
training-only spatially weighted RMS scales, with no mean subtraction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from .utils import sha256_file
else:
    from utils import sha256_file


INPUT_CHANNELS = ["E0_scaled", "H0_scaled", "alpha_scaled", "log_tau_scaled", "x_scaled"]
OUTPUT_CHANNELS = ["E_T_scaled", "H_T_scaled", "P_T_scaled"]
SPLITS = ("train", "validation", "test")
METHOD = "train_global_weighted_rms_no_center"


@dataclass
class PreparedData:
    X: np.ndarray
    Y: np.ndarray
    Y_physical: np.ndarray
    x: np.ndarray
    weights: np.ndarray
    split: np.ndarray
    sample_id: np.ndarray
    material_id: np.ndarray
    normalizer: dict[str, Any]
    fingerprints: dict[str, str]
    protocol: dict[str, Any]

    def indices(self, split_name: str) -> np.ndarray:
        if split_name not in SPLITS:
            raise ValueError(f"Unknown split: {split_name}")
        return np.flatnonzero(self.split == split_name)


def _finite(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued; imaginary components are not discarded")
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _tolerances(cfg: dict) -> dict[str, float]:
    data = cfg["data"]
    result = {key: float(data[key]) for key in ("time_atol", "zero_tolerance", "grid_atol")}
    result["material_atol"] = 1e-12
    if any(not np.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("Data tolerances must be finite and nonnegative")
    return result


def _grid(x, atol: float) -> np.ndarray:
    x = _finite(x, "x")
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a complete one-dimensional grid with at least two points")
    spacing = np.diff(x)
    if not np.all(spacing > 0):
        raise ValueError("x must be strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=0, atol=atol):
        raise ValueError("The FNO workflow requires a uniform spatial grid")
    return x


def _quadrature(x: np.ndarray) -> np.ndarray:
    widths = np.diff(x)
    weights = np.empty_like(x)
    weights[0], weights[-1] = widths[0] / 2, widths[-1] / 2
    weights[1:-1] = (widths[:-1] + widths[1:]) / 2
    return weights


def _ranges(raw_config: dict) -> dict[str, list[float]]:
    amin, amax = float(raw_config["alpha_min"]), float(raw_config["alpha_max"])
    tmin, tmax = float(raw_config["tau_min"]), float(raw_config["tau_max"])
    length = float(raw_config["L"])
    if not np.isfinite([amin, amax, tmin, tmax, length]).all():
        raise ValueError("Invalid physical parameter ranges")
    if not (0 < amin < amax < 1 and 0 < tmin < tmax and length > 0):
        raise ValueError("Invalid alpha/tau/domain ranges in the raw configuration")
    return {"alpha": [amin, amax], "log_tau": [float(np.log(tmin)), float(np.log(tmax))],
            "x": [0.0, length]}


def _split_sha256(sample_id, material_id, split) -> str:
    mapping = {"sample_id": np.asarray(sample_id).tolist(),
               "material_id": np.asarray(material_id).tolist(),
               "split": np.asarray(split).tolist()}
    payload = json.dumps(mapping, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_partition(sample_id, material_id, split, alpha, tau) -> None:
    count = len(alpha)
    for name, values in (("sample_id", sample_id), ("material_id", material_id), ("split", split)):
        if values.shape != (count,) or values.dtype.kind not in "US":
            raise ValueError(f"{name} must be a string vector with one entry per sample")
        if np.any(np.char.str_len(values.astype(str)) == 0):
            raise ValueError(f"Empty {name} entry")
    if len(np.unique(sample_id)) != count:
        raise ValueError("Duplicate sample IDs")
    if set(split.tolist()) != set(SPLITS):
        raise ValueError("The saved train/validation/test sets must all be nonempty")
    group_split, pair_split, group_pair, pair_group = {}, {}, {}, {}
    for group, part, a, t in zip(material_id.tolist(), split.tolist(), alpha, tau):
        pair = (float(a), float(t))
        if group in group_split and group_split[group] != part:
            raise ValueError("A material group crosses dataset splits")
        if pair in pair_split and pair_split[pair] != part:
            raise ValueError("A material parameter pair crosses dataset splits")
        if group in group_pair and group_pair[group] != pair:
            raise ValueError("A material ID names different parameter pairs")
        if pair in pair_group and pair_group[pair] != group:
            raise ValueError("One parameter pair has multiple material IDs")
        group_split[group], pair_split[pair] = part, part
        group_pair[group], pair_group[pair] = pair, group


def _check_material(alpha, tau, ranges, atol):
    alpha, tau = _finite(alpha, "alpha"), _finite(tau, "tau")
    if np.any((alpha <= 0) | (alpha >= 1)) or np.any(tau <= 0):
        raise ValueError("Require 0 < alpha < 1 and tau > 0")
    log_tau = np.log(tau)
    for name, values in (("alpha", alpha), ("log_tau", log_tau)):
        lo, hi = ranges[name]
        if np.any(values < lo - atol) or np.any(values > hi + atol):
            raise ValueError(f"{name} lies outside the prescribed material range; no clipping is applied")
    return alpha, tau


def _load_raw(cfg: dict) -> dict:
    settings = cfg["data"]
    target = float(settings["target_time"])
    tolerances = _tolerances(cfg)
    if not np.isfinite(target) or target != 1.0:
        raise ValueError("This experiment targets exactly T=1")
    raw_path, config_path = Path(settings["raw_path"]), Path(settings["raw_config_path"])
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    ranges = _ranges(raw_config)
    required = ("alpha", "tau", "initial_fields", "x", "times", "fields",
                "split", "sample_id", "material_id")
    with np.load(raw_path, allow_pickle=False) as saved:
        missing = set(required) - set(saved.files)
        if missing:
            raise ValueError(f"Raw dataset is missing fields: {sorted(missing)}")
        # Deliberately never read exact_fields, modal coefficients, or input coefficients.
        raw = {key: saved[key].copy() for key in required}
    x = _grid(raw["x"], tolerances["grid_atol"])
    if not np.allclose(x[[0, -1]], ranges["x"], atol=tolerances["grid_atol"], rtol=0):
        raise ValueError("Raw grid endpoints differ from the configured physical domain")
    times = _finite(raw["times"], "times")
    if times.ndim != 1 or times.size < 1 or not np.all(np.diff(times) > 0):
        raise ValueError("Raw times must be a unique, increasing vector")
    target_indices = np.flatnonzero(np.isclose(times, target, rtol=0, atol=tolerances["time_atol"]))
    zero_indices = np.flatnonzero(np.isclose(times, 0.0, rtol=0, atol=tolerances["time_atol"]))
    if len(target_indices) != 1 or len(zero_indices) != 1:
        raise ValueError("Raw data must contain exactly one T=1 frame and one initial frame")
    alpha, tau = _check_material(raw["alpha"], raw["tau"], ranges, tolerances["material_atol"])
    if alpha.ndim != 1 or tau.shape != alpha.shape or alpha.size == 0:
        raise ValueError("alpha and tau must be nonempty sample vectors")
    count = len(alpha)
    initial = _finite(raw["initial_fields"], "initial_fields")
    fields = _finite(raw["fields"], "fields")
    if initial.shape != (count, 3, len(x)) or fields.shape != (count, len(times), 3, len(x)):
        raise ValueError("Raw E/H/P channel order or array dimensions are inconsistent")
    if np.any(np.abs(initial[:, 2]) > tolerances["zero_tolerance"]):
        raise ValueError("This five-channel protocol requires P0=0")
    if np.any(np.abs(fields[:, int(zero_indices[0]), 2]) > tolerances["zero_tolerance"]):
        raise ValueError("The DG initial polarization is not zero")
    if np.any(np.abs(initial[:, 0, [0, -1]]) > tolerances["zero_tolerance"]):
        raise ValueError("The prescribed initial electric field violates PEC endpoint tolerance")
    for key in ("sample_id", "material_id", "split"):
        if raw[key].dtype.kind not in "US":
            raise ValueError(f"Raw {key} must contain strings")
        raw[key] = raw[key].astype(str)
    _validate_partition(raw["sample_id"], raw["material_id"], raw["split"], alpha, tau)
    return dict(alpha=alpha, tau=tau, initial_fields=initial,
                Y_physical=fields[:, int(target_indices[0])].copy(),
                x=x, weights=_quadrature(x), split=raw["split"], sample_id=raw["sample_id"],
                material_id=raw["material_id"], ranges=ranges, tolerances=tolerances,
                raw_sha256=sha256_file(raw_path), raw_config_sha256=sha256_file(config_path),
                split_sha256=_split_sha256(raw["sample_id"], raw["material_id"], raw["split"]),
                target_time=target)


def _training_scales(raw: dict) -> tuple[np.ndarray, np.ndarray]:
    train = raw["split"] == "train"
    weights = raw["weights"]
    def rms(values):
        result = np.sqrt(np.mean(np.sum(values[train]**2 * weights, axis=-1) / weights.sum(), axis=0))
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError("Training RMS scales must be strictly positive and finite")
        return result
    return rms(raw["initial_fields"][:, :2]), rms(raw["Y_physical"])


def validate_normalizer(norm: dict) -> None:
    if not isinstance(norm, dict) or type(norm.get("schema_version")) is not int or norm["schema_version"] != 1:
        raise ValueError("Unsupported normalizer schema")
    if norm.get("method") != METHOD or norm.get("fitted_on") != "train":
        raise ValueError("Only training-fitted global weighted RMS without centering is supported")
    if norm.get("input_channels") != INPUT_CHANNELS or norm.get("output_channels") != OUTPUT_CHANNELS:
        raise ValueError("Normalizer channel order is incompatible with this protocol")
    if float(norm.get("target_time", np.nan)) != 1.0:
        raise ValueError("Normalizer target_time must be 1")
    for key, count in (("input_scale", 2), ("output_scale", 3)):
        scale = _finite(norm[key], key)
        if scale.shape != (count,) or np.any(scale <= 0):
            raise ValueError(f"{key} has invalid shape or a zero scale")
    for key in ("raw_sha256", "split_sha256", "raw_config_sha256"):
        value = norm.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid {key}")
    tolerances = norm["tolerances"]
    for key in ("zero_tolerance", "grid_atol", "time_atol", "material_atol"):
        value = float(tolerances[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    ranges = norm["ranges"]
    for key in ("alpha", "log_tau", "x"):
        limits = _finite(ranges[key], key)
        if limits.shape != (2,) or limits[0] >= limits[1]:
            raise ValueError(f"Invalid {key} range")
    if not 0 < ranges["alpha"][0] < ranges["alpha"][1] < 1:
        raise ValueError("The alpha range must be strictly within (0,1)")
    if not np.isfinite(np.exp(ranges["log_tau"])).all() or np.any(np.exp(ranges["log_tau"]) <= 0):
        raise ValueError("The log-tau range does not represent positive finite tau")
    x = _grid(norm["x"], tolerances["grid_atol"])
    if not np.allclose(x[[0, -1]], ranges["x"], atol=tolerances["grid_atol"], rtol=0):
        raise ValueError("Normalizer grid and domain disagree")
    weights = _finite(norm["quadrature_weights"], "quadrature_weights")
    if weights.shape != x.shape or np.any(weights <= 0):
        raise ValueError("Quadrature weights must be positive and match x")
    if not np.allclose(weights, _quadrature(x), rtol=1e-12, atol=1e-14):
        raise ValueError("Expected physical-space trapezoidal quadrature weights")


def encode_inputs(alpha, tau, E0, H0, x, normalizer: dict) -> np.ndarray:
    validate_normalizer(normalizer)
    tol = normalizer["tolerances"]
    grid = _grid(x, tol["grid_atol"])
    expected_grid = np.asarray(normalizer["x"], dtype=np.float64)
    if grid.shape != expected_grid.shape or not np.allclose(grid, expected_grid, rtol=0, atol=tol["grid_atol"]):
        raise ValueError("Prediction requires the identical complete spatial grid")
    electric, magnetic = _finite(E0, "E0"), _finite(H0, "H0")
    if electric.ndim == 1:
        electric = electric[None]
    if magnetic.ndim == 1:
        magnetic = magnetic[None]
    if electric.ndim != 2 or magnetic.shape != electric.shape or electric.shape[1] != len(grid):
        raise ValueError("E0/H0 must both be [nx] or matching [batch,nx] arrays")
    batch = len(electric)
    if batch == 0:
        raise ValueError("Empty prediction batch")
    def material_batch(value, name):
        value = _finite(value, name)
        if value.ndim == 0:
            return np.full(batch, float(value))
        if value.shape != (batch,):
            raise ValueError(f"{name} must be a scalar or a vector of length batch")
        return value
    alpha, tau = _check_material(material_batch(alpha, "alpha"), material_batch(tau, "tau"),
                                normalizer["ranges"], tol["material_atol"])
    if np.any(np.abs(electric[:, [0, -1]]) > tol["zero_tolerance"]):
        raise ValueError(f"E0 endpoints must satisfy PEC within {tol['zero_tolerance']:g}; no correction is applied")
    def scaled(value, key):
        lo, hi = normalizer["ranges"][key]
        return 2 * (value - lo) / (hi - lo) - 1
    result = np.empty((batch, 5, len(grid)), dtype=np.float64)
    result[:, 0] = electric / normalizer["input_scale"][0]
    result[:, 1] = magnetic / normalizer["input_scale"][1]
    result[:, 2] = scaled(alpha, "alpha")[:, None]
    result[:, 3] = scaled(np.log(tau), "log_tau")[:, None]
    result[:, 4] = scaled(grid, "x")[None]
    encoded = result.astype(np.float32)
    if not np.isfinite(encoded).all():
        raise ValueError("Input cannot be represented as finite float32")
    return encoded


def decode_outputs(array, normalizer: dict) -> np.ndarray:
    validate_normalizer(normalizer)
    values = _finite(array, "model output")
    nx = len(normalizer["x"])
    if values.ndim not in (2, 3) or values.shape[-2:] != (3, nx):
        raise ValueError("Model output must be [3,nx] or [batch,3,nx]")
    result = values * np.asarray(normalizer["output_scale"], dtype=np.float64)[:, None]
    if not np.isfinite(result).all():
        raise ValueError("Decoded output is non-finite")
    return result


def protocol_from_normalizer(norm: dict) -> dict:
    validate_normalizer(norm)
    return dict(
        schema_version=1, target_time=float(norm["target_time"]),
        input_channels=list(INPUT_CHANNELS), output_channels=list(OUTPUT_CHANNELS),
        x=list(norm["x"]), parameter_ranges={key: list(value) for key, value in norm["ranges"].items()},
        field_normalization=METHOD, scalar_mapping="affine_to_minus_one_plus_one",
        initial_memory={"P0": 0.0, "history_before_t0": "zero"},
        initial_boundary={"E0": "PEC endpoints", "H0": "no zero-endpoint constraint",
                          "absolute_tolerance": float(norm["tolerances"]["zero_tolerance"])},
        grid_atol=float(norm["tolerances"]["grid_atol"]),
    )


def _roundtrip_equal(actual, expected, scale, name):
    # Float32 preparation rounds once; this checks decoding, not normalization.
    bound = 8 * np.finfo(np.float32).eps * np.maximum(np.abs(expected), scale)
    if actual.shape != expected.shape or np.any(np.abs(actual - expected) > bound):
        raise ValueError(f"Prepared {name} does not match its physical source")


def load_prepared(cfg: dict) -> PreparedData:
    settings = cfg["data"]
    prepared_path, norm_path = Path(settings["prepared_path"]), Path(settings["normalizer_path"])
    if not prepared_path.is_file() or not norm_path.is_file():
        raise FileNotFoundError("Prepared data are missing; explicitly run prepare_data.py first")
    normalizer = json.loads(norm_path.read_text(encoding="utf-8"))
    validate_normalizer(normalizer)
    norm_hash = sha256_file(norm_path)
    raw = _load_raw(cfg)
    for key in ("raw_sha256", "split_sha256", "raw_config_sha256"):
        if normalizer[key] != raw[key]:
            raise ValueError(f"Prepared normalizer {key} differs from the current raw data")
    if normalizer["ranges"] != raw["ranges"] or normalizer["tolerances"] != raw["tolerances"]:
        raise ValueError("Normalizer physical ranges/tolerances differ from the current protocol")
    input_scale, output_scale = _training_scales(raw)
    if not np.allclose(normalizer["input_scale"], input_scale, rtol=1e-12, atol=0) or not np.allclose(
            normalizer["output_scale"], output_scale, rtol=1e-12, atol=0):
        raise ValueError("Normalizer scales do not match training-only weighted RMS")
    required = ("X", "Y", "Y_physical", "x", "quadrature_weights", "split", "sample_id",
                "material_id", "target_time", "output_scale", "normalizer_json", "normalizer_sha256")
    with np.load(prepared_path, allow_pickle=False) as saved:
        if set(required) - set(saved.files):
            raise ValueError("Prepared dataset is missing required fields")
        saved = {key: saved[key].copy() for key in required}
    if saved["normalizer_json"].shape != () or json.loads(str(saved["normalizer_json"].item())) != normalizer:
        raise ValueError("Embedded normalizer differs from normalizer.json")
    if saved["normalizer_sha256"].shape != () or str(saved["normalizer_sha256"].item()) != norm_hash:
        raise ValueError("Embedded normalizer file hash mismatch")
    if saved["target_time"].shape != () or float(saved["target_time"]) != raw["target_time"]:
        raise ValueError("Prepared target time is not T=1")
    for key in ("x", "split", "sample_id", "material_id", "Y_physical"):
        if not np.array_equal(saved[key], raw[key]):
            raise ValueError(f"Prepared {key} differs from raw data")
    if not np.array_equal(saved["x"], np.asarray(normalizer["x"])) or not np.array_equal(
            saved["quadrature_weights"], raw["weights"]) or not np.array_equal(
            saved["quadrature_weights"], np.asarray(normalizer["quadrature_weights"])):
        raise ValueError("Prepared coordinates/quadrature disagree with normalizer or source")
    if not np.array_equal(saved["output_scale"], np.asarray(normalizer["output_scale"])):
        raise ValueError("Prepared output scale disagrees with normalizer")
    X, Y = saved["X"], saved["Y"]
    count, nx = len(raw["alpha"]), len(raw["x"])
    if X.shape != (count, 5, nx) or Y.shape != (count, 3, nx):
        raise ValueError("Prepared X/Y channel count or shape mismatch")
    if X.dtype != np.float32 or Y.dtype != np.float32 or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Prepared X/Y must already be finite float32 arrays")
    X64 = X.astype(np.float64)
    _roundtrip_equal(X64[:, :2] * input_scale[None, :, None], raw["initial_fields"][:, :2],
                     input_scale[None, :, None], "initial channels")
    for channel, key, values in ((2, "alpha", raw["alpha"]), (3, "log_tau", np.log(raw["tau"]))):
        lo, hi = normalizer["ranges"][key]
        decoded = (X64[:, channel] + 1) * ((hi - lo) / 2) + lo
        expected = np.broadcast_to(values[:, None], (count, nx))
        _roundtrip_equal(decoded, expected, hi - lo, key)
    lo, hi = normalizer["ranges"]["x"]
    grid = (X64[:, 4] + 1) * ((hi - lo) / 2) + lo
    _roundtrip_equal(grid, np.broadcast_to(raw["x"], (count, nx)), hi - lo, "coordinate channel")
    _roundtrip_equal(decode_outputs(Y, normalizer), saved["Y_physical"],
                     output_scale[None, :, None], "output decoding")
    fingerprints = {key: raw[key] for key in ("raw_sha256", "split_sha256", "raw_config_sha256")}
    fingerprints.update(prepared_sha256=sha256_file(prepared_path), normalizer_sha256=norm_hash)
    # Return stored X/Y unchanged: never apply a second normalization.
    return PreparedData(X=X, Y=Y, Y_physical=saved["Y_physical"], x=saved["x"],
                        weights=saved["quadrature_weights"], split=saved["split"],
                        sample_id=saved["sample_id"], material_id=saved["material_id"],
                        normalizer=normalizer, fingerprints=fingerprints,
                        protocol=protocol_from_normalizer(normalizer))
