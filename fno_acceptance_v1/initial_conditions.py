"""Deterministic, paired acceptance inputs; importing this module does no work.

The compact pulses are evaluated directly at arbitrary physical coordinates.
Their three-mode projections are diagnostics only and never replace an input.
No solver, model, file-writing, or prediction API is imported here.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _bump(s: np.ndarray, derivative: bool = False) -> np.ndarray:
    """b(s) or b'(s), with exact zero outside support and stable edge limits."""
    s = np.asarray(s, dtype=np.float64)
    result = np.zeros_like(s)
    inside = np.abs(s) < 1.0
    if derivative:
        inside &= s != 0.0
    u = s[inside]
    denominator = 1.0 - u * u
    exponent = 1.0 - 1.0 / denominator
    if derivative:
        # b'(s) = -2*s*b(s)/(1-s*s)^2. Combining logarithms avoids 0*inf.
        exponent = exponent + np.log(2.0 * np.abs(u)) - 2.0 * np.log(denominator)
        result[inside] = -np.sign(u) * np.exp(exponent)
    else:
        result[inside] = np.exp(exponent)
    return result


def _pulse_values(field: dict, z: np.ndarray) -> np.ndarray:
    values = np.zeros_like(z)
    derivative = bool(field["derivative"])
    for pulse in field["components"]:
        width = float(pulse["half_width"])
        center = float(pulse["center"])
        if not (0.0 < width and 0.0 < center - width < center + width < 1.0):
            raise ValueError("Pulse support must lie strictly inside 0 < x/L < 1")
        factor = float(pulse["weight"]) / (width if derivative else 1.0)
        values += factor * _bump((z - center) / width, derivative)
    return float(field["scale"]) * values


def evaluate_initial(spec: dict, x, L: float) -> np.ndarray:
    """Return E0/H0/P0 with shape ``(3,) + np.asarray(x).shape``.

    ``modal`` uses exactly the original three-mode continuous initial family.
    ``single_pulse``/``double_pulse`` use direct compact functions; H is the
    derivative of a compact potential with respect to z=x/L, not x.
    """
    coordinates = np.asarray(x)
    if np.iscomplexobj(coordinates):
        raise ValueError("Initial coordinates must be real")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not np.isfinite(coordinates).all() or not np.isfinite(L) or L <= 0:
        raise ValueError("Non-finite coordinates or invalid domain length")
    if np.any(coordinates < -1e-12 * L) or np.any(coordinates > L * (1 + 1e-12)):
        raise ValueError("Initial coordinates lie outside the physical domain")
    z = coordinates.reshape(-1) / L
    output = np.zeros((3, z.size), dtype=np.float64)
    family = spec["family"]
    if family == "modal":
        a, b = np.asarray(spec["a"], float), np.asarray(spec["b"], float)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
            raise ValueError("Modal initial conditions require three finite E/H coefficients")
        angles = np.arange(1, 4)[:, None] * np.pi * z[None, :]
        output[0] = a @ np.sin(angles)
        output[1] = b @ np.cos(angles)
        # These are the exact mathematical sine endpoint values, not an output
        # projection or a modification of a DG/FNO endpoint prediction.
        output[0, (z == 0.0) | (z == 1.0)] = 0.0
    elif family in ("single_pulse", "double_pulse"):
        if bool(spec["E"]["derivative"]) or not bool(spec["H"]["derivative"]):
            raise ValueError("Acceptance pulse protocol requires E=b and H=d(potential)/dz")
        output[0] = _pulse_values(spec["E"], z)
        output[1] = _pulse_values(spec["H"], z)
    else:
        raise ValueError(f"Unsupported initial family: {family}")
    if not np.isfinite(output).all():
        raise ValueError("Non-finite initial function value")
    return output.reshape((3,) + coordinates.shape)


def _quadrature(spec: dict, L: float, segments: int, order: int):
    """Composite Gauss-Legendre, explicitly split at all support edges."""
    breaks = list(np.linspace(0.0, 1.0, segments + 1))
    if spec["family"] != "modal":
        for name in ("E", "H"):
            for pulse in spec[name]["components"]:
                c, w = pulse["center"], pulse["half_width"]
                breaks.extend((c - w, c, c + w))
    breaks = np.unique(np.asarray(breaks, dtype=np.float64)) * L
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mids = (breaks[:-1] + breaks[1:]) / 2.0
    halves = np.diff(breaks) / 2.0
    return ((mids[:, None] + halves[:, None] * nodes).reshape(-1),
            (halves[:, None] * weights).reshape(-1))


def _continuous_diagnostics(spec: dict, L: float, settings: dict) -> dict:
    integrals = []
    for order in (settings["quadrature_high_order"], settings["quadrature_low_order"]):
        points, weights = _quadrature(spec, L, settings["quadrature_segments"], order)
        values = evaluate_initial(spec, points, L)
        norms = np.sqrt(np.sum(values * values * weights[None, :], axis=1))
        integrals.append((points, weights, values, norms, np.sum(values * weights[None, :], axis=1)))
    points, weights, values, norms, means = integrals[0]
    result = {
        "norm_E": float(norms[0]), "norm_H": float(norms[1]),
        "H_continuous_integral": float(means[1]),
        "H_continuous_integral_low_order": float(integrals[1][4][1]),
        "H_integral_quadrature_difference": float(abs(means[1] - integrals[1][4][1])),
        "quadrature_norm_difference_E": float(abs(norms[0] - integrals[1][3][0])),
        "quadrature_norm_difference_H": float(abs(norms[1] - integrals[1][3][1])),
        "quadrature": {"segments": settings["quadrature_segments"],
                       "high_order": settings["quadrature_high_order"],
                       "low_order": settings["quadrature_low_order"],
                       "support_edges_split": True},
    }
    angles = np.arange(1, 4)[:, None] * np.pi * points[None, :] / L
    for index, (field, basis) in enumerate((("E", np.sin(angles)), ("H", np.cos(angles)))):
        coefficients = (2.0 / L) * (basis @ (weights * values[index]))
        residual = values[index] - coefficients @ basis
        result[f"q_{field}"] = (float(np.sqrt(np.sum(weights * residual ** 2)) / norms[index])
                                 if norms[index] > 0.0 else None)
        result[f"diagnostic_projection_coefficients_{field}"] = coefficients.tolist()
    return result


def _check_settings(settings: dict) -> None:
    required = (
        "seed", "material_count", "initials_per_material", "new_alpha_range", "new_tau_range",
        "material_max_attempts", "material_duplicate_atol", "hull_tolerance", "center_range",
        "half_width_range", "double_min_center_separation", "double_secondary_weight_range",
        "minimum_support_grid_intervals", "minimum_cancellation_ratio", "amplitude_quantiles",
        "quadrature_segments", "quadrature_high_order", "quadrature_low_order",
        "quadrature_relative_tolerance", "quadrature_absolute_tolerance",
        "zero_mean_absolute_tolerance", "input_duplicate_atol", "q_near_modal_threshold",
    )
    missing = sorted(set(required) - set(settings))
    if missing:
        raise ValueError(f"Missing explicit design settings: {missing}")
    if settings["material_count"] != 4 or settings["initials_per_material"] != 4:
        raise ValueError("Acceptance v1 has four materials and four initials per group")
    if settings["seed"] != 20260925:
        raise ValueError("Acceptance v1 freezes its new input seed at 20260925")
    if settings["amplitude_quantiles"] != [0.25, 0.75]:
        raise ValueError("Acceptance v1 amplitude design uses train initial Q25--Q75")
    for name in ("new_alpha_range", "new_tau_range", "center_range", "half_width_range",
                 "double_secondary_weight_range"):
        limits = np.asarray(settings[name], dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f"Invalid range {name}")
    if settings["new_tau_range"][0] <= 0 or settings["half_width_range"][0] <= 0:
        raise ValueError("Tau and pulse half widths must be positive")
    if settings["center_range"][0] - settings["half_width_range"][1] <= 0 or (
            settings["center_range"][1] + settings["half_width_range"][1] >= 1):
        raise ValueError("Every planned pulse support must lie strictly inside the domain")
    if not 0 < settings["double_min_center_separation"] < np.diff(settings["center_range"])[0]:
        raise ValueError("Invalid double-pulse minimum center separation")
    for name in ("material_max_attempts", "quadrature_segments", "quadrature_high_order", "quadrature_low_order"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings["quadrature_high_order"] <= settings["quadrature_low_order"]:
        raise ValueError("High-order quadrature must exceed low-order quadrature")
    for name in ("material_duplicate_atol", "hull_tolerance", "minimum_support_grid_intervals",
                 "minimum_cancellation_ratio", "quadrature_relative_tolerance", "quadrature_absolute_tolerance",
                 "zero_mean_absolute_tolerance", "input_duplicate_atol", "q_near_modal_threshold"):
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _material_design(raw: dict, normalizer: dict, settings: dict, rng) -> tuple:
    from scipy.spatial import ConvexHull

    ids = raw["material_id"].astype(str)
    split = raw["split"].astype(str)
    unique = sorted(set(ids.tolist()))
    records = []
    for mid in unique:
        indices = np.flatnonzero(ids == mid)
        i = int(indices[0])
        if len(set(split[indices])) != 1 or not np.all(raw["alpha"][indices] == raw["alpha"][i]) or not np.all(
                raw["tau"][indices] == raw["tau"][i]):
            raise ValueError(f"Inconsistent material metadata or cross-split material: {mid}")
        records.append({"material_id": mid, "alpha": float(raw["alpha"][i]), "tau": float(raw["tau"][i]),
                        "source_split": str(split[i])})
    training = [item for item in records if item["source_split"] == "train"]
    if len(training) < 4:
        raise ValueError("Fewer than four distinct training materials")
    origin = np.array([normalizer["ranges"]["alpha"][0], normalizer["ranges"]["log_tau"][0]])
    span = np.array([np.diff(normalizer["ranges"]["alpha"])[0], np.diff(normalizer["ranges"]["log_tau"])[0]])
    coordinates = np.array([[r["alpha"], np.log(r["tau"])] for r in training])
    scaled = (coordinates - origin) / span
    # Stable lower-left anchor, then maximin farthest-point coverage. Ties use
    # lexicographic material_id because the records were sorted before selection.
    chosen = [int(np.argmin(np.sum(scaled, axis=1)))]
    while len(chosen) < 4:
        distances = np.min(np.sum((scaled[:, None] - scaled[chosen][None]) ** 2, axis=2), axis=1)
        distances[chosen] = -1.0
        chosen.append(int(np.argmax(distances)))
    seen = [copy.deepcopy(training[i]) for i in chosen]
    hull = ConvexHull(scaled)
    all_coordinates = np.array([[r["alpha"], np.log(r["tau"])] for r in records])
    alpha_edges = np.linspace(*settings["new_alpha_range"], 3)
    log_edges = np.linspace(*np.log(settings["new_tau_range"]), 3)
    if alpha_edges[0] < origin[0] or alpha_edges[-1] > origin[0] + span[0] or (
            log_edges[0] < origin[1] or log_edges[-1] > origin[1] + span[1]):
        raise ValueError("New material stratum lies outside the frozen normalization ranges")
    new = []
    for ai in range(2):
        for ti in range(2):
            for attempt in range(1, settings["material_max_attempts"] + 1):
                point = np.array([rng.uniform(alpha_edges[ai], alpha_edges[ai + 1]),
                                  rng.uniform(log_edges[ti], log_edges[ti + 1])])
                duplicate_points = all_coordinates if not new else np.vstack((all_coordinates,
                    [[row["alpha"], np.log(row["tau"])] for row in new]))
                duplicate = np.any(np.max(np.abs(duplicate_points - point), axis=1) <= settings["material_duplicate_atol"])
                value = (point - origin) / span
                inside = bool(np.all(hull.equations[:, :2] @ value + hull.equations[:, 2] <= settings["hull_tolerance"]))
                if inside and not duplicate:
                    new.append({"material_id": f"acceptance_new_material_{len(new):02d}",
                                "alpha": float(point[0]), "tau": float(np.exp(point[1])),
                                "source_split": "new_acceptance", "stratum": [ai, ti],
                                "coordinate_attempts": attempt, "inside_training_hull": True})
                    break
            else:
                raise ValueError("Fixed material-only candidate budget exhausted; no prediction-based resampling is allowed")
    information = {"seen_rule": "sorted material IDs; minimum scaled alpha+logtau anchor, then maximin Euclidean distance",
                   "new_rule": "one uniform alpha/logtau draw in each 2x2 stratum; reject duplicate/outside train hull",
                   "coordinate_origin": origin.tolist(), "coordinate_span": span.tolist(),
                   "training_hull_equations": hull.equations.tolist(),
                   "training_hull_vertices_material_id": [training[i]["material_id"] for i in hull.vertices],
                   "original_material_count": len(records), "training_material_count": len(training)}
    return seen, new, information


def _new_pulse_field(rng, count: int, derivative: bool, settings: dict) -> dict:
    lo, hi = settings["center_range"]
    if count == 1:
        centers = [float(rng.uniform(lo, hi))]
    else:
        separation = settings["double_min_center_separation"]
        first = float(rng.uniform(lo, hi - separation))
        centers = [first, float(rng.uniform(first + separation, hi))]
    components = []
    for index, center in enumerate(centers):
        magnitude = 1.0 if index == 0 else float(rng.uniform(*settings["double_secondary_weight_range"]))
        weight = float(rng.choice([-1.0, 1.0])) * magnitude
        components.append({"center": center, "half_width": float(rng.uniform(*settings["half_width_range"])),
                           "weight": weight})
    return {"components": components, "scale": 1.0, "derivative": derivative}


def _build_pulse(rng, count: int, targets: dict, L: float, settings: dict) -> tuple:
    spec = {"family": "single_pulse" if count == 1 else "double_pulse",
            "E": _new_pulse_field(rng, count, False, settings),
            "H": _new_pulse_field(rng, count, True, settings)}
    points, weights = _quadrature(spec, L, settings["quadrature_segments"], settings["quadrature_high_order"])
    values = evaluate_initial(spec, points, L)
    desired = {}
    cancellation = {}
    for index, name in enumerate(("E", "H")):
        raw_norm = float(np.sqrt(np.sum(weights * values[index] ** 2)))
        desired[name] = float(rng.uniform(*targets[name]))
        component_norm_sum = 0.0
        for component in spec[name]["components"]:
            single = {"components": [component], "scale": 1.0, "derivative": spec[name]["derivative"]}
            component_values = _pulse_values(single, points / L)
            component_norm_sum += float(np.sqrt(np.sum(weights * component_values ** 2)))
        cancellation[name] = raw_norm / component_norm_sum if component_norm_sum > 0 else 0.0
        # A zero/near-cancelling draw remains recorded and is invalidated below;
        # it is never replaced after looking at the function or model output.
        spec[name]["scale"] = desired[name] / raw_norm if raw_norm > 0 else 1.0
    return spec, desired, cancellation


def build_design(cfg: dict, project_root, normalizer: dict) -> tuple[dict, dict]:
    """Build the frozen 64-case design in memory when explicitly called.

    The caller persists the manifest, config and hashes before solving/predicting.
    Invalid input draws remain present with ``INPUT_INVALID`` and an issue list.
    """
    settings = copy.deepcopy(cfg["design"])
    _check_settings(settings)
    project_root = Path(project_root)
    dataset_path = project_root / "data" / "dataset.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        required = ("alpha", "tau", "x", "sample_id", "material_id", "split", "initial_fields")
        raw = {name: data[name].copy() for name in required}
    x = np.asarray(normalizer["x"], dtype=np.float64)
    weights = np.asarray(normalizer["quadrature_weights"], dtype=np.float64)
    L = float(normalizer["ranges"]["x"][1])
    if normalizer["ranges"]["x"][0] != 0 or x.shape != (65,) or not np.array_equal(x, raw["x"]):
        raise ValueError("Acceptance v1 must use the original 65-point [0,L] interface")
    if raw["initial_fields"].shape != (len(raw["alpha"]), 3, 65):
        raise ValueError("Unexpected original initial_fields layout")
    for name in ("alpha", "tau", "initial_fields"):
        if not np.isfinite(raw[name]).all():
            raise ValueError(f"Non-finite original {name}")
    if np.any(raw["tau"] <= 0) or len(set(raw["sample_id"].astype(str))) != len(raw["alpha"]):
        raise ValueError("Invalid original tau or duplicate sample IDs")
    train_indices = np.flatnonzero(raw["split"].astype(str) == "train")
    if len(train_indices) < 2:
        raise ValueError("Two original training samples are required for interface regression")
    norms = np.sqrt(np.sum(raw["initial_fields"][train_indices, :2] ** 2 * weights[None, None], axis=2))
    targets = {name: np.quantile(norms[:, i], settings["amplitude_quantiles"]).tolist() for i, name in enumerate(("E", "H"))}
    rng = np.random.default_rng(settings["seed"])
    seen, new, material_information = _material_design(raw, normalizer, settings, rng)
    modal_specs = [{"family": "modal", "a": (rng.uniform(-1, 1, 3) / np.arange(1, 4) ** 2).tolist(),
                    "b": (rng.uniform(-1, 1, 3) / np.arange(1, 4) ** 2).tolist()} for _ in range(16)]
    pulses = [_build_pulse(rng, 1 if index % 4 < 2 else 2, targets, L, settings) for index in range(16)]
    cases, samples = [], []
    original_initial_hashes = [_array_hash(row) for row in raw["initial_fields"]]
    original_full_hashes = [_array_hash(np.array([a, t]), row) for a, t, row in
                            zip(raw["alpha"], raw["tau"], raw["initial_fields"])]
    input_atol = settings["input_duplicate_atol"]
    for group in ("G1", "G2", "G3", "G4"):
        materials = seen if group in ("G1", "G3") else new
        for index in range(16):
            material = materials[index // 4]
            if group in ("G1", "G2"):
                spec, desired, cancellation = copy.deepcopy(modal_specs[index]), {}, {}
            else:
                spec, desired, cancellation = copy.deepcopy(pulses[index])
            initial = evaluate_initial(spec, x, L)
            diagnostics = _continuous_diagnostics(spec, L, settings)
            issues, warnings = [], []
            diagnostics.update({"target_norm_E": desired.get("E"), "target_norm_H": desired.get("H"),
                "scale_E": 1.0 if spec["family"] == "modal" else spec["E"]["scale"],
                "scale_H": 1.0 if spec["family"] == "modal" else spec["H"]["scale"],
                "E_endpoints": initial[0, [0, -1]].tolist(), "P_max_abs": float(np.max(np.abs(initial[2]))),
                "H_nodal_integral": float(weights @ initial[1]),
                "H_nodal_minus_continuous_integral": float(weights @ initial[1] - diagnostics["H_continuous_integral"]),
                "nodal_norm_E": float(np.sqrt(weights @ initial[0] ** 2)),
                "nodal_norm_H": float(np.sqrt(weights @ initial[1] ** 2)),
                "sampled_initial_sha256": _array_hash(initial),
                "full_input_sha256": _array_hash(np.array([material["alpha"], material["tau"]]), initial),
                "initial_spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
            if not np.isfinite(initial).all():
                issues.append("nonfinite_initial")
            if np.any(np.abs(initial[0, [0, -1]]) > settings["quadrature_absolute_tolerance"]) or np.any(initial[2] != 0):
                issues.append("PEC_E_or_P0_incompatibility")
            if abs(diagnostics["H_continuous_integral"]) > settings["zero_mean_absolute_tolerance"]:
                issues.append("H_continuous_mean_not_zero")
            for field in ("E", "H"):
                tol = settings["quadrature_absolute_tolerance"] + settings["quadrature_relative_tolerance"] * diagnostics[f"norm_{field}"]
                if diagnostics[f"quadrature_norm_difference_{field}"] > tol:
                    issues.append(f"{field}_initial_quadrature_not_converged")
                if field in desired and abs(diagnostics[f"norm_{field}"] - desired[field]) > tol:
                    issues.append(f"{field}_initial_amplitude_scaling_mismatch")
            if spec["family"] != "modal":
                grid_dz = float(np.max(np.diff(x)) / L)
                supports = [2 * p["half_width"] / grid_dz for f in ("E", "H") for p in spec[f]["components"]]
                diagnostics["minimum_support_grid_intervals"] = float(min(supports))
                diagnostics["cancellation_ratio"] = cancellation
                if min(supports) < settings["minimum_support_grid_intervals"]:
                    issues.append("pulse_support_below_fixed_grid_resolution_rule")
                if min(cancellation.values()) < settings["minimum_cancellation_ratio"]:
                    issues.append("near_cancelling_pulses")
                if all(diagnostics[f"q_{f}"] is not None and diagnostics[f"q_{f}"] < settings["q_near_modal_threshold"] for f in ("E", "H")):
                    warnings.append("function_family_change_limited_by_q_diagnostic")
            old_initial_matches = np.flatnonzero(np.all(np.abs(raw["initial_fields"] - initial) <= input_atol, axis=(1, 2)))
            diagnostics["old_initial_matches"] = raw["sample_id"][old_initial_matches].astype(str).tolist()
            if len(old_initial_matches):
                issues.append("initial_samples_duplicate_an_original_sample")
            old_full_matches = [i for i in old_initial_matches if abs(raw["alpha"][i] - material["alpha"]) <= settings["material_duplicate_atol"]
                                and abs(np.log(raw["tau"][i]) - np.log(material["tau"])) <= settings["material_duplicate_atol"]]
            diagnostics["old_full_input_matches"] = raw["sample_id"][old_full_matches].astype(str).tolist()
            if old_full_matches:
                issues.append("full_input_duplicates_original_trajectory")
            for previous, sample in zip(cases, samples):
                if np.all(np.abs(initial - sample) <= input_atol):
                    paired = (previous["pair_index"] == index and
                              frozenset((previous["group"], group)) in (frozenset(("G1", "G2")), frozenset(("G3", "G4"))))
                    if not paired:
                        issues.append(f"unexpected_acceptance_initial_duplicate:{previous['case_id']}")
                    if abs(previous["alpha"] - material["alpha"]) <= settings["material_duplicate_atol"] and abs(
                            np.log(previous["tau"]) - np.log(material["tau"])) <= settings["material_duplicate_atol"]:
                        issues.append(f"acceptance_full_input_duplicate:{previous['case_id']}")
            diagnostics["issues"], diagnostics["warnings"] = issues, warnings
            cases.append({"case_id": f"{group}_{index:02d}", "group": group, "family": spec["family"],
                          "material_id": material["material_id"], "alpha": material["alpha"], "tau": material["tau"],
                          "log_tau": float(np.log(material["tau"])), "pair_index": index,
                          "initial_spec": spec, "status": "INPUT_INVALID" if issues else "VALID",
                          "input_diagnostics": diagnostics})
            samples.append(initial)
    # Modal representatives cover the first and last preselected material. New
    # function representatives are the narrowest single and double before any
    # solver or predictor has been called. Ties use pair_index deterministically.
    narrowest = []
    for family in ("single_pulse", "double_pulse"):
        candidates = [case for case in cases if case["group"] == "G3" and case["family"] == family]
        pick = min(candidates, key=lambda c: (min(p["half_width"] for f in ("E", "H")
                   for p in c["initial_spec"][f]["components"]), c["pair_index"]))
        narrowest.append(pick["pair_index"])
    representatives = [f"{group}_{index:02d}" for group in ("G1", "G2", "G3", "G4")
                       for index in ((0, 15) if group in ("G1", "G2") else narrowest)]
    manifest = {
        "schema_version": 1, "seed": settings["seed"], "design": settings,
        "planned_case_count": 64, "cases_per_group": 16, "materials_per_group": 4,
        "task": {"T": 1.0, "L": L, "P0": 0.0, "initial_history": "fixed_zero", "boundary": "PEC",
                 "x_points": 65, "input_fields": "continuous_initial_samples"},
        "materials": {"seen": seen, "new": new}, "material_selection": material_information,
        "training_initial_norm_quantiles": targets,
        "amplitude_target_definition": "Train E0/H0 physical L2 Q25--Q75 on original trapezoidal grid; pulse scale from continuous composite Gauss norms",
        "q_definition": "Continuous L2 projection onto sine m=1..3 for E and cosine m=1..3 for H; diagnostic only",
        "original_input_fingerprints": [{"sample_id": str(sid), "sampled_initial_sha256": ih, "full_input_sha256": fh}
                                         for sid, ih, fh in zip(raw["sample_id"], original_initial_hashes, original_full_hashes)],
        "regression_sample_ids": raw["sample_id"][train_indices[:2]].astype(str).tolist(),
        "representative_case_ids": representatives, "refinement_case_ids": representatives,
        "representative_selection": "G1/G2 pair indices 0 and 15; G3/G4 minimum pulse half-width separately for single/double, before predictions",
        "pairing": {"G2_initial_equals": "G1 by pair_index", "G4_initial_equals": "G3 by pair_index"},
        "cases": cases,
        "limitations": ["Four distinct materials per group; 16 cases are not 16 independent material draws",
                        "Compact H derivative/zero mean controls the initial family; it is not a PEC requirement",
                        "q does not validate arbitrary high frequencies or arbitrary new initial functions"],
    }
    arrays = {"initial_fields": np.stack(samples), "x": x.copy(), "case_id": np.array([c["case_id"] for c in cases])}
    manifest["arrays_sha256"] = {"initial_fields": _array_hash(arrays["initial_fields"]), "x": _array_hash(x)}
    return manifest, arrays
