"""Frozen, endpoint-only acceptance metrics and user-triggered reporting.

Importing this module performs no inference, DG solve, timing, file write or plot.
All errors below are 65-node trapezoidal diagnostics at T=1, not rigorous
continuous error bounds.  Missing/nonfinite cases remain in the planned counts.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = ("E", "H", "P")
GROUPS = ("G1", "G2", "G3", "G4")
DEFAULT_ACCEPTANCE = {
    "relative_zero_factor": 1e-10,
    "weak_field_factor": 0.1,
    "field_mean_relative_limit": 0.02,
    "joint_relative_limit": 0.05,
    "joint_min_fraction": 0.9,
    "planned_per_group": 16,
    "thresholds": [0.01, 0.02, 0.05],
    "refinement_abs_factor": 1e-4,
    "refinement_rel_factor": 0.002,
}
GROUP_DESCRIPTIONS = {
    "G1": "训练见过的材料；原三模态分布中新抽初值",
    "G2": "新的范围内材料；与 G1 配对的相同初值",
    "G3": "与 G1 相同材料；新的紧支撑平滑初值",
    "G4": "与 G2 相同新材料；与 G3 配对的相同初值",
}


def _json_safe(value: Any) -> Any:
    """Use JSON null, never nonstandard NaN/Infinity, for unavailable metrics."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _dump_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False)


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summary_stats(values: list[Any]) -> dict[str, Any]:
    valid = [number for item in values if (number := _finite_number(item)) is not None]
    if not valid:
        return {"mean": None, "median": None, "p95": None, "max": None,
                "defined_count": 0, "missing_count": len(values)}
    array = np.asarray(valid, dtype=np.float64)
    return {"mean": float(np.mean(array)), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(np.max(array)),
            "defined_count": len(valid), "missing_count": len(values) - len(valid)}


def _acceptance(cfg: dict) -> dict:
    settings = dict(DEFAULT_ACCEPTANCE)
    settings.update(cfg.get("acceptance", {}))
    if settings["planned_per_group"] != 16:
        raise ValueError("This frozen v1 protocol requires 16 planned cases per group")
    thresholds = [float(value) for value in settings["thresholds"]]
    if thresholds != [0.01, 0.02, 0.05]:
        raise ValueError("v1 must retain the predeclared 1%, 2%, 5% counts")
    for name in ("relative_zero_factor", "weak_field_factor",
                 "field_mean_relative_limit", "joint_relative_limit",
                 "refinement_abs_factor", "refinement_rel_factor"):
        number = _finite_number(settings[name])
        if number is None or number <= 0:
            raise ValueError(f"Invalid acceptance setting: {name}")
        settings[name] = number
    if settings["weak_field_factor"] <= settings["relative_zero_factor"]:
        raise ValueError("weak_field_factor must exceed relative_zero_factor")
    fraction = _finite_number(settings["joint_min_fraction"])
    if fraction is None or not 0 < fraction <= 1:
        raise ValueError("joint_min_fraction must be in (0,1]")
    settings["joint_min_fraction"] = fraction
    settings["thresholds"] = thresholds
    return settings


def _grid_and_scales(normalizer: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if _finite_number(normalizer.get("target_time")) != 1.0:
        raise ValueError("Acceptance v1 evaluates only the frozen target T=1")
    x = np.asarray(normalizer["x"], dtype=np.float64)
    scales = np.asarray(normalizer["output_scale"], dtype=np.float64)
    if x.shape != (65,) or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Metrics require the original 65 strictly increasing nodes")
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Frozen output_scale must contain three finite positive values")
    length = float(x[-1] - x[0])
    steps = np.diff(x)
    weights = np.empty_like(x)
    weights[0], weights[-1] = steps[0] / 2, steps[-1] / 2
    weights[1:-1] = (steps[:-1] + steps[1:]) / 2
    if "quadrature_weights" in normalizer:
        original = np.asarray(normalizer["quadrature_weights"], dtype=np.float64)
        if original.shape != weights.shape or not np.allclose(
            original, weights, rtol=1e-12, atol=1e-14
        ):
            raise ValueError("Frozen quadrature weights disagree with 65-node trapezoidal weights")
    return x, weights, scales, length


def _weighted_norm(values: np.ndarray, weights: np.ndarray) -> float | None:
    if not np.isfinite(values).all():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.sqrt(np.sum(np.square(values) * weights))
    return _finite_number(result)


def refinement_metrics(reference, finer, weights, scales, L,
                       abs_factor=1e-4, rel_factor=0.002) -> dict:
    """Compare numerical refinements per field, without claiming a true-error bound.

    Both arrays are [3,nx].  Output is JSON-safe, with ``fields.E/H/P`` holding
    abs_diff, norm_finer, threshold, pass and finite, and a top-level all_pass.
    """
    reference = np.asarray(reference, dtype=np.float64)
    finer = np.asarray(finer, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] != 3 or finer.shape != reference.shape:
        raise ValueError("Refinement arrays must both have shape [3,nx]")
    if weights.shape != (reference.shape[1],) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Refinement quadrature weights must be finite and positive")
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Refinement output scales must be finite and positive")
    if not math.isfinite(float(L)) or L <= 0 or not np.isclose(weights.sum(), L, rtol=1e-10):
        raise ValueError("L must agree with the total physical quadrature weight")
    if not (math.isfinite(float(abs_factor)) and math.isfinite(float(rel_factor))
            and abs_factor >= 0 and rel_factor >= 0):
        raise ValueError("Refinement factors must be finite and nonnegative")
    fields = {}
    for index, field in enumerate(FIELDS):
        norm_finer = _weighted_norm(finer[index], weights)
        with np.errstate(over="ignore", invalid="ignore"):
            diff = _weighted_norm(finer[index] - reference[index], weights)
        threshold = (float(abs_factor * scales[index] * math.sqrt(L) + rel_factor * norm_finer)
                     if norm_finer is not None else None)
        finite = diff is not None and threshold is not None and math.isfinite(threshold)
        fields[field] = {"abs_diff": diff, "norm_finer": norm_finer,
                         "threshold": threshold, "pass": bool(finite and diff <= threshold),
                         "finite": bool(finite)}
    return _json_safe({"fields": fields, "all_pass": all(item["pass"] for item in fields.values()),
                       "finite": all(item["finite"] for item in fields.values()),
                       "abs_factor": abs_factor, "rel_factor": rel_factor,
                       "criterion": "norm(finer-reference) <= abs_factor*s_U*sqrt(L) + rel_factor*norm(finer)",
                       "interpretation": "numerical self-consistency only; not a rigorous error bound"})


def _flatten_metadata(value: dict, prefix: str = "case") -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}_{key}"
        if isinstance(item, dict):
            result.update(_flatten_metadata(item, name))
        elif isinstance(item, (list, tuple, np.ndarray)):
            result[name] = _dump_json(item)
        else:
            result[name] = _json_safe(item)
    return result


def _interface_state(reference_checks: dict) -> tuple[bool, dict]:
    interface = reference_checks.get("interface", {})
    states = {name: interface.get(name, {}).get("status", "NOT_RUN")
              for name in ("dg_regression", "fno_regression")}
    return all(status == "PASS" for status in states.values()), states


def _reference_status(check: dict, group_check: dict, finite: bool) -> str:
    if not finite:
        return "INCOMPLETE"
    status = str(check.get("status", "NOT_RECORDED"))
    if status == "INCOMPLETE" or any(word in status.upper() for word in ("FAIL", "MISSING", "ERROR")):
        return "INCOMPLETE"
    if status in ("REFERENCE_UNCERTAIN", "UNCERTAIN", "NEEDS_REVIEW"):
        return "REFERENCE_UNCERTAIN"
    group_status = group_check.get("status", "INCOMPLETE")
    if group_status == "SAMPLED_PASS" and status != "NOT_RECORDED":
        return "SAMPLED_PASS"
    if group_status == "REFERENCE_UNCERTAIN":
        return "REFERENCE_UNCERTAIN"
    return "INCOMPLETE"


def evaluate_cases(manifest, initials, predictions, references,
                   reference_checks, normalizer, cfg) -> tuple[list[dict], dict]:
    """Evaluate every manifest case, retaining failed rows and planned denominators.

    Arrays are aligned to manifest['cases'] and have shape [case,3,65].  NaN
    placeholders for missing cases are allowed.  Descriptive error summaries
    include finite data even when its reference is uncertain; acceptance counts
    additionally require valid input, passed interfaces and sampled reference
    checks.  These two count types are never silently conflated.
    """
    settings = _acceptance(cfg)
    x, weights, scales, length = _grid_and_scales(normalizer)
    cases = manifest["cases"]
    ids = [str(case["case_id"]) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case_id in frozen manifest")
    for case in cases:
        if case["group"] not in GROUPS:
            raise ValueError(f"Unknown group: {case['group']}")
    expected = (len(cases), 3, len(x))
    initials = np.asarray(initials, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if initials.shape != expected or predictions.shape != expected or references.shape != expected:
        raise ValueError(f"Initial/prediction/reference arrays must all have shape {expected}")
    interface_pass, interface_states = _interface_state(reference_checks)
    case_checks = reference_checks.get("cases", {})
    group_checks = reference_checks.get("groups", {})
    root_length = math.sqrt(length)
    rows = []
    for index, case in enumerate(cases):
        case_id, group = ids[index], case["group"]
        initial, prediction, reference = initials[index], predictions[index], references[index]
        diagnostic = case.get("input_diagnostics", {})
        check = case_checks.get(case_id, {})
        prediction_record = reference_checks.get("prediction_status", {}).get(case_id, {})
        prediction_record_status = (str(prediction_record.get("status", "NOT_RECORDED"))
                                    if isinstance(prediction_record, dict) else str(prediction_record))
        prediction_record_failure = any(word in prediction_record_status.upper()
                                        for word in ("FAIL", "ERROR", "MISSING", "INCOMPLETE", "NOT_RUN"))
        pred_finite = bool(np.isfinite(prediction).all())
        ref_finite = bool(np.isfinite(reference).all())
        initial_finite = bool(np.isfinite(initial).all())
        input_valid = case.get("status") == "VALID" and initial_finite
        ref_state = _reference_status(check, group_checks.get(group, {}), ref_finite)
        eligible = (input_valid and pred_finite and not prediction_record_failure
                    and ref_state == "SAMPLED_PASS" and interface_pass)
        row = {
            "case_id": case_id, "group": group, "family": case.get("family"),
            "material_id": case.get("material_id"), "alpha": case.get("alpha"),
            "tau": case.get("tau"), "pair_index": case.get("pair_index"),
            "input_status": case.get("status", "NOT_RECORDED"), "input_valid": input_valid,
            "prediction_finite": pred_finite, "reference_finite": ref_finite,
            "prediction_record_status": prediction_record_status,
            "prediction_record_failure": prediction_record_failure,
            "reference_status": ref_state, "reference_case_status": check.get("status", "NOT_RECORDED"),
            "interface_pass": interface_pass, "eligible_for_acceptance": eligible,
            "q_E": diagnostic.get("q_E"), "q_H": diagnostic.get("q_H"),
            "initial_E_continuous_norm": diagnostic.get("norm_E"),
            "initial_H_continuous_norm": diagnostic.get("norm_H"),
            "initial_H_continuous_integral": diagnostic.get("H_continuous_integral"),
            "baseline_vs_reference_json": _dump_json(check.get("baseline_vs_reference")),
            "space_refinement_json": _dump_json(check.get("space")),
            "time_refinement_json": _dump_json(check.get("time")),
            "reference_check_json": _dump_json(check),
            "manifest_case_json": _dump_json(case),
        }
        row.update(_flatten_metadata(case))
        for field_index, field in enumerate(FIELDS):
            initial_norm = _weighted_norm(initial[field_index], weights)
            norm = _weighted_norm(reference[field_index], weights)
            with np.errstate(over="ignore", invalid="ignore"):
                difference = prediction[field_index] - reference[field_index]
                absolute = _weighted_norm(difference, weights)
            scale_norm = float(scales[field_index] * root_length)
            zero_threshold = settings["relative_zero_factor"] * scale_norm
            weak_threshold = settings["weak_field_factor"] * scale_norm
            defined = norm is not None and absolute is not None and norm > zero_threshold
            relative = (absolute / norm) if defined else None
            relative = _finite_number(relative)
            weak = norm is not None and norm < weak_threshold
            row.update({
                f"initial_{field}_nodal_norm": initial_norm,
                f"{field}_norm_ref": norm, f"{field}_abs_l2": absolute,
                f"{field}_relative_l2": relative,
                f"{field}_scaled_abs_l2": absolute / scale_norm if absolute is not None else None,
                f"{field}_max_point_abs": (float(np.max(np.abs(difference)))
                                            if np.isfinite(difference).all() else None),
                f"{field}_relative_defined": relative is not None,
                f"{field}_numerical_zero": norm is not None and norm <= zero_threshold,
                f"{field}_weak_field": bool(weak),
                f"{field}_zero_threshold": zero_threshold,
                f"{field}_weak_threshold": weak_threshold,
            })
        relatives = [row[f"{field}_relative_l2"] for field in FIELDS]
        row["R"] = max(relatives) if all(value is not None for value in relatives) else None
        row["relative_undefined_fields"] = ",".join(
            field for field in FIELDS if row[f"{field}_relative_l2"] is None)
        for threshold in settings["thresholds"]:
            key = f"joint_le_{threshold:g}"
            row[key] = row["R"] is not None and row["R"] <= threshold
            row[f"{key}_eligible"] = bool(row[key] and eligible)
        for endpoint, endpoint_index in (("left", 0), ("right", -1)):
            predicted_endpoint = _finite_number(prediction[0, endpoint_index])
            reference_endpoint = _finite_number(reference[0, endpoint_index])
            row[f"E_{endpoint}_prediction"] = predicted_endpoint
            row[f"E_{endpoint}_reference_interior_trace"] = reference_endpoint
            row[f"E_{endpoint}_prediction_minus_reference"] = (
                predicted_endpoint - reference_endpoint
                if predicted_endpoint is not None and reference_endpoint is not None else None)
        for label, values in (("initial", initial), ("prediction", prediction), ("reference", reference)):
            row[f"H_{label}_nodal_integral"] = (
                float(np.dot(weights, values[1])) if np.isfinite(values[1]).all() else None)
        for label in ("prediction", "reference"):
            terminal = row[f"H_{label}_nodal_integral"]
            nodal_initial = row["H_initial_nodal_integral"]
            continuous_initial = _finite_number(row["initial_H_continuous_integral"])
            row[f"H_{label}_minus_initial_nodal_integral"] = (
                terminal - nodal_initial if terminal is not None and nodal_initial is not None else None)
            row[f"H_{label}_nodal_minus_initial_continuous_integral"] = (
                terminal - continuous_initial if terminal is not None and continuous_initial is not None else None)
        if not pred_finite or not ref_finite or prediction_record_failure:
            row["case_status"] = "INCOMPLETE"
        elif not eligible or row["R"] is None:
            row["case_status"] = "NEEDS_REVIEW"
        elif row["R"] <= settings["joint_relative_limit"]:
            row["case_status"] = "WITHIN_JOINT_THRESHOLD"
        else:
            row["case_status"] = "ABOVE_JOINT_THRESHOLD"
        rows.append(_json_safe(row))

    summary = {
        "schema_version": 1,
        "protocol": {
            "target_time": 1.0, "node_count": len(x), "L": length,
            "quadrature": "65-node physical trapezoidal weights",
            "fields": list(FIELDS), "output_scales": scales.tolist(),
            "acceptance": settings, "planned_cases": 64,
            "sampled_reference_checks_are_not_all_case_error_bounds": True,
            "descriptive_statistics_include_uncertain_reference_if_finite": True,
            "undefined_relatives_are_not_replaced_by_zero": True,
            "thresholds_are_working_choices_not_industry_standards": True,
        },
        "interface": interface_states, "groups": {},
        "manual_review_required": True,
    }
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        planned = int(settings["planned_per_group"])
        if len(selected) > planned:
            raise ValueError(f"{group} contains more than its frozen {planned} planned cases")
        reference_group = group_checks.get(group, {})
        group_summary = {
            "description": GROUP_DESCRIPTIONS[group], "planned_count": planned,
            "present_count": len(selected), "actual_material_count": len({row["material_id"] for row in selected}),
            "material_ids": sorted({str(row["material_id"]) for row in selected}),
            "reference_status": reference_group.get("status", "INCOMPLETE"),
            "checked_case_ids": reference_group.get("checked_case_ids", []),
            "fields": {}, "R": _summary_stats([row["R"] for row in selected]),
            "threshold_counts": {},
            "counts": {
                "missing_manifest_cases": planned - len(selected),
                "nonfinite_prediction": sum(not row["prediction_finite"] for row in selected),
                "prediction_record_failure": sum(row["prediction_record_failure"] for row in selected),
                "nonfinite_reference": sum(not row["reference_finite"] for row in selected),
                "input_unresolved": sum(not row["input_valid"] for row in selected),
                "reference_uncertain": sum(row["reference_status"] == "REFERENCE_UNCERTAIN" for row in selected),
                "reference_incomplete": sum(row["reference_status"] == "INCOMPLETE" for row in selected),
                "relative_undefined": sum(row["R"] is None for row in selected),
                "eligible_for_acceptance": sum(row["eligible_for_acceptance"] and row["R"] is not None for row in selected),
                "weak_field_cases": sum(any(row[f"{field}_weak_field"] for field in FIELDS) for row in selected),
            },
            "q_E": _summary_stats([row["q_E"] for row in selected]),
            "q_H": _summary_stats([row["q_H"] for row in selected]),
            "manual_review_required": True,
            "manual_review_scope": "All weak fields and the worst case; no automatic deletion or replacement",
        }
        q_threshold = _finite_number(cfg.get("design", {}).get("q_near_modal_threshold"))
        near_modal = [row["case_id"] for row in selected
                      if q_threshold is not None and row["q_E"] is not None and row["q_H"] is not None
                      and row["q_E"] < q_threshold and row["q_H"] < q_threshold]
        group_summary["function_family_diagnostic"] = {
            "q_near_modal_threshold": q_threshold,
            "near_modal_case_ids": near_modal,
            "near_modal_count": len(near_modal),
            "limited_family_change": group in ("G3", "G4") and len(near_modal) > planned / 2,
            "is_acceptance_filter": False,
        }
        alphas = [_finite_number(row["alpha"]) for row in selected]
        taus = [_finite_number(row["tau"]) for row in selected]
        group_summary["actual_alpha_range"] = ([min(alphas), max(alphas)]
                                               if alphas and all(value is not None for value in alphas) else None)
        group_summary["actual_tau_range"] = ([min(taus), max(taus)]
                                             if taus and all(value is not None for value in taus) else None)
        for field in FIELDS:
            field_summary = _summary_stats([row[f"{field}_relative_l2"] for row in selected])
            weak_rows = [row for row in selected if row[f"{field}_weak_field"]]
            field_summary.update({
                "absolute_l2": _summary_stats([row[f"{field}_abs_l2"] for row in selected]),
                "scaled_absolute_l2": _summary_stats([row[f"{field}_scaled_abs_l2"] for row in selected]),
                "reference_norm": _summary_stats([row[f"{field}_norm_ref"] for row in selected]),
                "max_point_absolute": _summary_stats([row[f"{field}_max_point_abs"] for row in selected]),
                "weak_field_count": len(weak_rows),
                "weak_field_case_ids": [row["case_id"] for row in weak_rows],
                "weak_field_abs_l2": _summary_stats([row[f"{field}_abs_l2"] for row in weak_rows]),
                "numerical_zero_count": sum(row[f"{field}_numerical_zero"] for row in selected),
            })
            group_summary["fields"][field] = field_summary
        for threshold in settings["thresholds"]:
            observed = sum(row["R"] is not None and row["R"] <= threshold for row in selected)
            eligible_count = sum(row["R"] is not None and row["R"] <= threshold
                                 and row["eligible_for_acceptance"] for row in selected)
            group_summary["threshold_counts"][f"{threshold:g}"] = {
                "passed": eligible_count, "observed_below_threshold": observed,
                "planned": planned, "relative_defined": sum(row["R"] is not None for row in selected),
                "unresolved_or_missing": planned - group_summary["counts"]["eligible_for_acceptance"],
            }
        finite_r = [row for row in selected if row["R"] is not None]
        worst = max(finite_r, key=lambda row: row["R"]) if finite_r else None
        group_summary["worst_case_id"] = worst["case_id"] if worst else None
        group_summary["unresolved_case_ids"] = [
            row["case_id"] for row in selected if row["case_status"] in ("INCOMPLETE", "NEEDS_REVIEW")]
        # Even if R is undefined, an absolute-error plot must remain available.
        absolute_candidates = [row for row in selected
                               if any(row[f"{field}_scaled_abs_l2"] is not None for field in FIELDS)]
        worst_absolute = (max(absolute_candidates, key=lambda row: max(
            value for field in FIELDS if (value := row[f"{field}_scaled_abs_l2"]) is not None))
                          if absolute_candidates else None)
        group_summary["worst_scaled_absolute_case_id"] = worst_absolute["case_id"] if worst_absolute else None
        reasons = []
        counts = group_summary["counts"]
        incomplete = (counts["missing_manifest_cases"] or counts["nonfinite_prediction"] or
                      counts["prediction_record_failure"] or
                      counts["nonfinite_reference"] or counts["reference_incomplete"] or
                      group_summary["reference_status"] == "INCOMPLETE")
        review = (not interface_pass or counts["input_unresolved"] or counts["relative_undefined"] or
                  counts["reference_uncertain"] or group_summary["reference_status"] != "SAMPLED_PASS")
        for name, count in counts.items():
            if count and name not in ("eligible_for_acceptance", "weak_field_cases"):
                reasons.append(f"{name}={count}")
        if not interface_pass:
            reasons.append("DG/FNO interface regression is not PASS")
        if group_summary["reference_status"] != "SAMPLED_PASS":
            reasons.append(f"group reference status={group_summary['reference_status']}")
        numerical_gate = "UNRESOLVED"
        if not incomplete and not review:
            means_ok = all(group_summary["fields"][field]["mean"] is not None and
                           group_summary["fields"][field]["mean"] <= settings["field_mean_relative_limit"]
                           for field in FIELDS)
            joint_count = sum(row["R"] is not None and row["R"] <= settings["joint_relative_limit"]
                              and row["eligible_for_acceptance"] for row in selected)
            required_count = math.ceil(planned * settings["joint_min_fraction"] - 1e-12)
            numerical_gate = "PASS" if means_ok and joint_count >= required_count else "FAIL"
            if not means_ok:
                reasons.append("At least one field mean exceeds its predeclared relative-error limit")
            if joint_count < required_count:
                reasons.append(f"Joint threshold count {joint_count}/{planned}; requires {required_count}/{planned}")
        group_summary["numerical_gate"] = numerical_gate
        group_summary["status"] = ("INCOMPLETE" if incomplete else "NEEDS_REVIEW" if review else
                                   "PRELIMINARY_PASS" if numerical_gate == "PASS" else "PRELIMINARY_FAIL")
        group_summary["reasons"] = reasons
        summary["groups"][group] = group_summary
    return _json_safe(rows), _json_safe(summary)


def _fmt(value, *, percent=False) -> str:
    number = _finite_number(value)
    if number is None:
        return "未定义/待运行"
    return f"{100 * number:.3g}%" if percent else f"{number:.5g}"


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _recorded_speedup(benchmark: dict, group: str, configuration: str,
                      cache: str, scope: str) -> float | None:
    """Publish only a ratio approved and stored by the benchmark producer.

    Finite partial medians are descriptive data, not permission to derive a
    replacement ratio.  Per-regime status permits a complete group/regime to
    remain reportable when another regime failed; global PARTIAL is not itself
    a reason to manufacture or to discard a locally validated speedup.
    """
    if scope not in ("wrapper", "solve"):
        return None
    record = benchmark.get("groups", {}).get(group, {})
    fno = record.get("fno", {})
    regime = record.get("dg", {}).get(configuration, {}).get(cache, {})
    if fno.get("status") != "COMPLETE" or regime.get("status") != "COMPLETE":
        return None
    protocol = benchmark.get("timing_protocol", {})
    warmups = _finite_number(protocol.get("fno_warmups"))
    successful_warmups = _finite_number(protocol.get("fno_successful_warmups"))
    fno_count = _finite_number(fno.get("count"))
    if (protocol.get("fno_warmup_status") != "COMPLETE" or warmups is None or warmups < 20
            or successful_warmups != warmups or fno_count is None or fno_count < 200):
        return None
    fno_median = _finite_number(fno.get("median_seconds"))
    dg_median = _finite_number(regime.get(scope, {}).get("median_seconds"))
    if fno_median is None or fno_median <= 0 or dg_median is None or dg_median <= 0:
        return None
    key = f"{configuration}_{cache}_{scope}_over_fno"
    stored = _finite_number(record.get("speedups", {}).get(key))
    return stored if stored is not None and stored > 0 else None


def _timing_cell(benchmark: dict, group: str) -> str:
    record = benchmark.get("groups", {}).get(group, {})
    fno = record.get("fno", {}).get("median_seconds")
    dg = record.get("dg", {}).get("baseline", {}).get("cache_hit", {}).get("wrapper", {}).get("median_seconds")
    fno_number, dg_number = _finite_number(fno), _finite_number(dg)
    if fno_number is None and dg_number is None:
        return "待计时；不提供倍率"
    ratio = _recorded_speedup(benchmark, group, "baseline", "cache_hit", "wrapper")
    ratio_text = (f"S={ratio:.3g}" if ratio is not None else
                  "计时不完整或未验证；不提供倍率")
    return (f"FNO {_fmt(fno_number)} s；DG {_fmt(dg_number)} s；{ratio_text}；"
            "基线 cache_hit wrapper/FNO 常驻 batch=1")


def _interpretation(summary: dict) -> list[str]:
    states = {group: summary["groups"][group]["status"] for group in GROUPS}
    gates = {group: summary["groups"][group]["numerical_gate"] for group in GROUPS}
    if any(state in ("INCOMPLETE", "NEEDS_REVIEW") for state in states.values()):
        return ["存在缺失、参考质量、接口或未定义相对误差问题；相应组结论暂缓，不能直接归因于 FNO。"]
    if gates["G1"] == "FAIL":
        return ["G1 未满足数值门槛：先排查冻结权重加载、归一化、初值接口、参考求解与原分布表现；本工具不启动重训。"]
    observations = []
    if all(gates[group] == "PASS" for group in ("G1", "G2")) and any(
        gates[group] == "FAIL" for group in ("G3", "G4")
    ):
        observations.append("原三模态初值族表现较好，但新函数族尚未满足门槛；不能以延长旧分布训练替代新函数覆盖。")
    if all(gates[group] == "PASS" for group in ("G1", "G3")) and any(
        gates[group] == "FAIL" for group in ("G2", "G4")
    ):
        observations.append("见过材料的组表现较好，新材料组未全部满足门槛；优先排查参数覆盖与参数泛化。")
    if all(gate == "PASS" for gate in gates.values()):
        observations.append("四组均初步满足预设数值门槛；弱场与最差样本仍需人工复核。速度须按各声明口径另行判断。")
    return observations or ["按各组及各场结果界定当前使用范围；该 64 条配对样本不构成总体成功率或全函数空间认证。"]


def _plot_cases(output_dir, manifest, arrays, rows, summary) -> dict:
    """Plot only on explicit write_report; unavailable cases are annotated, not dropped."""
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError as error:
        return {"status": "INCOMPLETE", "reason": str(error), "files": [],
                "note": "Plot dependency missing; install manually then rerun report. No samples replaced."}
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    indices = {str(case["case_id"]): index for index, case in enumerate(manifest["cases"])}
    by_id = {row["case_id"]: row for row in rows}
    representatives = manifest.get("representative_case_ids", [])
    if isinstance(representatives, dict):
        representatives = [case_id for values in representatives.values()
                           for case_id in (values if isinstance(values, list) else [values])]
    selections = {str(case_id): ["predeclared_representative"] for case_id in representatives}
    for group in GROUPS:
        record = summary["groups"][group]
        for key, reason in (("worst_case_id", "largest_defined_R"),
                            ("worst_scaled_absolute_case_id", "largest_scaled_absolute_error")):
            if record.get(key) is not None:
                selections.setdefault(record[key], []).append(reason)
        # Missing/undefined cases can be absent from any numerical ranking.
        for case_id in record["unresolved_case_ids"]:
            selections.setdefault(case_id, []).append("unresolved_case_preserved")
    x = np.asarray(arrays["x"], dtype=np.float64)
    predictions = np.asarray(arrays["predictions"], dtype=np.float64)
    references = np.asarray(arrays["references"], dtype=np.float64)
    files, errors = [], []
    for case_id, reasons in selections.items():
        if case_id not in indices:
            errors.append({"case_id": case_id, "reason": "Predeclared case is absent from manifest"})
            continue
        row = by_id[case_id]
        index = indices[case_id]
        for field_index, field in enumerate(FIELDS):
            figure = Figure(figsize=(9, 6), constrained_layout=True)
            FigureCanvasAgg(figure)
            upper, lower = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
            predicted, reference = predictions[index, field_index], references[index, field_index]
            if np.isfinite(reference).all():
                upper.plot(x, reference, color="black", label="DG numerical reference (T=1)")
            else:
                upper.text(0.02, 0.92, "DG reference missing/nonfinite", transform=upper.transAxes, color="red")
            if np.isfinite(predicted).all():
                upper.plot(x, predicted, "--", color="#1769aa", label="Frozen FNO (T=1)")
            else:
                upper.text(0.02, 0.8, "FNO prediction missing/nonfinite", transform=upper.transAxes, color="red")
            if np.isfinite(predicted).all() and np.isfinite(reference).all():
                lower.plot(x, np.abs(predicted - reference), color="#b44b15")
            else:
                lower.text(0.02, 0.6, "Error unavailable; case remains in the planned denominator", transform=lower.transAxes)
            if upper.lines:
                upper.legend(loc="best")
            upper.set_ylabel(f"{field} (physical scale)")
            lower.set_ylabel("Pointwise abs. error")
            lower.set_xlabel("x")
            upper.grid(alpha=0.2)
            lower.grid(alpha=0.2)
            title = (f"{case_id} / {field}, alpha={row['alpha']}, tau={row['tau']}\n"
                     f"rel={_fmt(row[f'{field}_relative_l2'], percent=True)}, "
                     f"abs={_fmt(row[f'{field}_abs_l2'])}, weak={row[f'{field}_weak_field']}, "
                     f"reference={row['reference_status']}")
            figure.suptitle(title)
            filename = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{case_id}_{field}.png")
            figure.savefig(plot_dir / filename, dpi=160)
            files.append({"case_id": case_id, "field": field, "reasons": reasons,
                          "path": f"plots/{filename}"})
            figure.clear()
    return {"status": "COMPLETE" if not errors else "INCOMPLETE", "files": files, "errors": errors,
            "selection": "All preregistered representatives, largest R, largest scaled absolute error, and unresolved cases"}


def _report_markdown(manifest, rows, summary, reference_checks, benchmark, cfg) -> str:
    lines = ["# 冻结 FNO：T=1 四组初步验收", "",
             "|组别与范围|实际材料/计划样本|E/H/P 平均相对误差|三场 ≤1% / 2% / 5% 观察数（可判数）|R P95 / 最大|弱场绝对误差最大 E/H/P|参考检查|时间与倍率口径|初步判定|",
             "|---|---|---|---|---|---|---|---|---|"]
    for group in GROUPS:
        record = summary["groups"][group]
        mean_errors = " / ".join(_fmt(record["fields"][field]["mean"], percent=True) for field in FIELDS)
        counts = " / ".join(
            f"{record['threshold_counts'][f'{threshold:g}']['observed_below_threshold']}/16"
            f"（{record['threshold_counts'][f'{threshold:g}']['passed']}）"
            for threshold in (0.01, 0.02, 0.05))
        weak = " / ".join(_fmt(record["fields"][field]["weak_field_abs_l2"]["max"])
                          if record["fields"][field]["weak_field_count"] else "无弱场" for field in FIELDS)
        r_tail = f"{_fmt(record['R']['p95'], percent=True)} / {_fmt(record['R']['max'], percent=True)}"
        status = record["status"]
        if status == "PRELIMINARY_PASS":
            status = "数值门槛初步满足；人工复核待完成"
        elif status == "PRELIMINARY_FAIL":
            status = "未满足本轮数值门槛"
        values = [f"{group}：{record['description']}", f"{record['actual_material_count']} / 16",
                  mean_errors, counts, r_tail, weak, record["reference_status"],
                  _timing_cell(benchmark, group), status]
        lines.append("|" + "|".join(_cell(value) for value in values) + "|")
    lines.extend(["", "以上观察数分母始终为每组全部计划的 16 条，括号内是其中输入合法、接口通过、参考抽样检查通过且三场相对误差均定义的可判通过数。参考不确定时，即使观察误差较小也不算验收通过；也不能把可判数为零理解为 FNO 误差全都超标。描述性均值保留有限数据，不能覆盖参考状态。",
                  "", "S=DG/FNO；S>1 表示相对所列基线更快，S<1 表示该口径不占优。主表 DG 包含当前 Python 封装/进程/文件 IO；这不是双方相同进程内算法耗时的证明。",
                  "", "初判为本轮工作阈值，不是行业标准。每场平均相对误差 ≤"
                  f"{_fmt(summary['protocol']['acceptance']['field_mean_relative_limit'], percent=True)}，"
                  f"且至少 {math.ceil(16 * summary['protocol']['acceptance']['joint_min_fraction'] - 1e-12)}/16 条三场同时 ≤"
                  f"{_fmt(summary['protocol']['acceptance']['joint_relative_limit'], percent=True)}，"
                  "并要求无未解决缺失、参考或接口问题。弱场和最差样本须人工复核。",
                  "", "## 结论边界", ""])
    lines.extend(f"- {item}" for item in _interpretation(summary))
    lines.extend([
        "- 每组默认 4 个材料、每材料 4 条初值；配对样本不是独立材料重复。此处计数不是未来总体成功率。",
        "- 新材料仅取本轮 alpha∈[0.35,0.65]、tau∈[0.65,1.6] 的训练凸包内设计；不扩张到整个原参数矩形或边缘。实际坐标保留在 CSV 和 manifest。",
        "- 新函数为本轮宽而平滑的紧支撑脉冲。H 的零均值是控制变量选择；其他 H 均值/常数模态、高频、尖峰和不相容边界不在已验范围。",
        "- 能表示函数空间中的算子、能输入新波形、在训练外波形上实测准确，是三个不同命题。前两个不能替代第三个。",
        "- 仅有 T=1 输出；没有验证全时段 Caputo 残差、逐时刻稳定性、完整能量耗散或 DG 通量误差。DG 细化是数值自一致性检查，不是精确解或严格误差上界。",
        "", "## 分组明细与待复核项", "",
    ])
    for group in GROUPS:
        record = summary["groups"][group]
        lines.extend([f"### {group}", "",
                      f"状态 `{record['status']}`；数值门槛 `{record['numerical_gate']}`；实际材料数 {record['actual_material_count']}；实际 alpha 范围 {_dump_json(record['actual_alpha_range'])}，tau 范围 {_dump_json(record['actual_tau_range'])}。",
                      f"参考预定检查样本：{', '.join(record['checked_case_ids']) or '未记录'}。",
                      f"R 最大样本：{record['worst_case_id'] or '未定义'}；缩放绝对误差最大样本：{record['worst_scaled_absolute_case_id'] or '未定义'}。",
                      "", "|场|mean|median|P95|max|有定义数/计划数|弱场数|绝对误差最大|参考范数最小|",
                      "|---|---|---|---|---|---|---|---|---|"])
        for field in FIELDS:
            stats = record["fields"][field]
            norms = [row[f"{field}_norm_ref"] for row in rows if row["group"] == group
                     and row[f"{field}_norm_ref"] is not None]
            values = [field] + [_fmt(stats[key], percent=True) for key in ("mean", "median", "p95", "max")]
            values += [f"{stats['defined_count']}/16", stats["weak_field_count"],
                       _fmt(stats["absolute_l2"]["max"]), _fmt(min(norms) if norms else None)]
            lines.append("|" + "|".join(_cell(value) for value in values) + "|")
        lines.extend(["", f"计数：`{_dump_json(record['counts'])}`。",
                      f"未定义/不确定/缺失 ID：{', '.join(record['unresolved_case_ids']) or '无'}。",
                      f"原因：{'; '.join(record['reasons']) or '数值门槛之外，弱场/最差样本仍待人工复核'}。",
                      f"q_E 中位/P95：{_fmt(record['q_E']['median'])} / {_fmt(record['q_E']['p95'])}；q_H 中位/P95：{_fmt(record['q_H']['median'])} / {_fmt(record['q_H']['p95'])}。仅据此展示离三模态空间的距离，不认证任意高频。",
                      "", "弱场、未定义误差与最大误差样本（不删除）：", "",
                      "|case|场|reference state|norm_ref|abs_l2|真实 relative|scaled absolute|weak|",
                      "|---|---|---|---|---|---|---|---|"])
        for row in rows:
            if row["group"] != group:
                continue
            for field in FIELDS:
                if (row[f"{field}_weak_field"] or row[f"{field}_relative_l2"] is None or
                    row["case_id"] in (record["worst_case_id"], record["worst_scaled_absolute_case_id"])):
                    values = [row["case_id"], field, row["reference_status"], _fmt(row[f"{field}_norm_ref"]),
                              _fmt(row[f"{field}_abs_l2"]), _fmt(row[f"{field}_relative_l2"], percent=True),
                              _fmt(row[f"{field}_scaled_abs_l2"]), row[f"{field}_weak_field"]]
                    lines.append("|" + "|".join(_cell(value) for value in values) + "|")
        if record["function_family_diagnostic"]["limited_family_change"]:
            lines.extend(["", "**函数族变化程度有限**：该组超过一半计划样本的 E/H 初值距原三模态空间的相对距离均低于预设诊断阈值。仍保留全部样本；不能据此声称已验证任意高频或任意初值。"])
        lines.append("")
    lines.extend(["## 计时完整口径", "",
                  f"计时状态：`{benchmark.get('status', 'NOT_RUN')}`。冷启动：`{_dump_json(benchmark.get('cold_start', {}))}`。",
                  "", "FNO 为常驻 batch=1，包含编码、tensor 转换、前向与反归一化。DG solve 与 wrapper 分列；缓存命中状态以实际记录为准，LU/分解是否重建以 timing_protocol 为准。编译、散列、画图和保存结果不计入查询延迟。",
                  "", "只发布 benchmark 已保存且 FNO/对应 DG 计时均完整的倍率。预热不足、计时样本不足、部分求解失败或缓存状态不符时，有限时长仍展示，但不以中位数重新计算倍率。",
                  "", "|组|设置|缓存|口径|median s|P95 s|max s|次数|计时状态|已验证倍率|",
                  "|---|---|---|---|---|---|---|---|---|---|"])
    for group in GROUPS:
        timing = benchmark.get("groups", {}).get(group, {})
        fno = timing.get("fno", {})
        values = [group, "FNO", "resident", "batch=1 end-to-end"]
        values += [_fmt(fno.get(key)) for key in ("median_seconds", "p95_seconds", "max_seconds")]
        values += [fno.get("count", 0), fno.get("status", "NOT_RUN"), "—"]
        lines.append("|" + "|".join(_cell(value) for value in values) + "|")
        for configuration in ("baseline", "reference"):
            for cache in ("cache_hit", "cache_miss"):
                cache_record = timing.get("dg", {}).get(configuration, {}).get(cache, {})
                for scope in ("wrapper", "solve", "projection"):
                    stats = cache_record.get(scope, {})
                    ratio = _recorded_speedup(benchmark, group, configuration, cache, scope)
                    values = [group, configuration, cache, scope]
                    values += [_fmt(stats.get(key)) for key in ("median_seconds", "p95_seconds", "max_seconds")]
                    ratio_text = (_fmt(ratio) if ratio is not None else "不完整/未验证；不提供倍率")
                    values += [stats.get("count", 0), cache_record.get("status", "NOT_RUN"),
                               ratio_text if scope != "projection" else "不单独称加速"]
                    lines.append("|" + "|".join(_cell(value) for value in values) + "|")
    lines.extend(["", "基线/参考设置与计时环境完整记录：", "", "```json",
                  json.dumps(_json_safe({"configured_dg": cfg.get("dg", cfg.get("reference", {})),
                                        "timing_protocol": benchmark.get("timing_protocol", {})}),
                             ensure_ascii=False, indent=2, allow_nan=False), "```", "",
                  "不宣称同精度最优 DG 也比 FNO 慢。基线与参考的逐样本差异、空间与时间分开细化结果均保留在 per_case_metrics.csv/reference_checks.json。",
                  "", "## 端点与积分诊断", "",
                  "E 仅比较网络端点与 DG 内侧迹，未强制修改预测。H 积分分别保留初始连续积分、65 点梯形积分、终点预测/参考梯形积分；其差异包含节点求积误差，不能直接解释为连续守恒定理失效。", "",
                  "|case|E左端差|E右端差|H初始连续积分|H初始节点积分|H预测末积分|H参考末积分|", "|---|---|---|---|---|---|---|"])
    for row in rows:
        values = [row["case_id"]] + [_fmt(row[key]) for key in (
            "E_left_prediction_minus_reference", "E_right_prediction_minus_reference",
            "initial_H_continuous_integral", "H_initial_nodal_integral",
            "H_prediction_nodal_integral", "H_reference_nodal_integral")]
        lines.append("|" + "|".join(_cell(value) for value in values) + "|")
    lines.extend(["", "## 图与产物", "",
                  f"绘图状态：`{summary.get('plots', {}).get('status', 'NOT_RUN')}`。预固定代表样本、R 最大样本、缩放绝对误差最大样本及未解决样本均保留；每场单独绘图。",
                  "", "全部 64 条计划样本的字段、ID、初值参数、q、状态和数值检查见 per_case_metrics.csv、cases_manifest.json 与 reference_checks.json；predictions_and_reference.npz 保留未修改的物理尺度数组，失败用 NaN。",
                  "", f"接口检查：`{_dump_json(reference_checks.get('interface', {}))}`。", ""])
    for figure in summary.get("plots", {}).get("files", []):
        lines.extend([f"![{figure['case_id']} {figure['field']}]({figure['path']})", ""])
    if summary.get("plots", {}).get("status") != "COMPLETE":
        lines.extend([f"图未完整产生：`{_dump_json(summary.get('plots', {}))}`。", ""])
    return "\n".join(lines)


def write_report(output_dir, manifest, arrays, reference_checks,
                 benchmark, normalizer, cfg) -> dict:
    """Write real supplied artifacts only, when explicitly called by the user-run pipeline.

    Required arrays: initial_fields, predictions, references, x. Optional
    prediction_status is a dict keyed by case_id and is preserved in the CSV/JSON.
    No calls to a model, solver, subprocess, trainer or timer occur here.
    """
    output_dir = Path(output_dir)
    x, weights, _, _ = _grid_and_scales(normalizer)
    supplied_x = np.asarray(arrays["x"], dtype=np.float64)
    if supplied_x.shape != x.shape or not np.array_equal(supplied_x, x):
        raise ValueError("Report coordinates must exactly match the frozen normalizer grid")
    prediction_status = arrays.get("prediction_status", {})
    checks_for_evaluation = dict(reference_checks)
    if isinstance(prediction_status, dict):
        checks_for_evaluation["prediction_status"] = prediction_status
    rows, summary = evaluate_cases(manifest, arrays["initial_fields"], arrays["predictions"],
                                   arrays["references"], checks_for_evaluation, normalizer, cfg)
    for row in rows:
        status = prediction_status.get(row["case_id"], {}) if isinstance(prediction_status, dict) else {}
        row["prediction_status_json"] = _dump_json(status)
    summary["prediction_status"] = _json_safe(prediction_status)
    summary["benchmark"] = _json_safe(benchmark)
    summary["reference_checks"] = _json_safe(reference_checks)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "predictions_and_reference.npz",
        initial_fields=np.asarray(arrays["initial_fields"], dtype=np.float64),
        predictions=np.asarray(arrays["predictions"], dtype=np.float64),
        references=np.asarray(arrays["references"], dtype=np.float64), x=x, weights=weights,
        case_id=np.asarray([str(case["case_id"]) for case in manifest["cases"]], dtype=str),
        group=np.asarray([str(case["group"]) for case in manifest["cases"]], dtype=str),
        target_time=np.asarray(1.0),
        prediction_status_json=np.asarray(_dump_json(prediction_status)),
    )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "per_case_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary["plots"] = _plot_cases(output_dir, manifest, arrays, rows, summary)
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT_验收结论.md").write_text(
        _report_markdown(manifest, rows, summary, reference_checks, benchmark, cfg), encoding="utf-8")
    return summary
