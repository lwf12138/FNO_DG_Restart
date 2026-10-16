"""Manual frozen-A acceptance. Default all; importing never runs any stage."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

if __package__:
    from .acceptance_common import (HERE, canonical, implementation_hashes, load_configuration,
        read_json, require, sha256, track, verify_artifact, verify_sources, write_json)
else:
    from acceptance_common import (HERE, canonical, implementation_hashes, load_configuration,
        read_json, require, sha256, track, verify_artifact, verify_sources, write_json)

GROUPS = ("G1", "G2", "G3", "G4")


def write_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def protocol_identity(cfg, paths):
    return {"config_sha256": hashlib.sha256(canonical(cfg).encode()).hexdigest(),
            "sources": verify_sources(cfg, paths), "implementation": implementation_hashes()}


def load_context(cfg, paths):
    import numpy as np
    output = paths["output"]
    require((output / "protocol.json").is_file(), "Run --stage prepare first; no implicit resampling")
    protocol = read_json(verify_artifact(output, "protocol.json"))
    require(protocol["identity"] == protocol_identity(cfg, paths),
            "Frozen config/source/implementation changed; preserve this run and use a separately reviewed new output directory")
    manifest = read_json(verify_artifact(output, "cases_manifest.json"))
    with np.load(verify_artifact(output, "initial_fields.npz"), allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    require(len(manifest["cases"]) == 64 and arrays["initial_fields"].shape == (64, 3, 65), "Incomplete prepared design")
    require(arrays["case_id"].tolist() == [c["case_id"] for c in manifest["cases"]], "Case ordering differs")
    normalizer = read_json(paths["checkpoint"].parent / "normalizer.json")
    return manifest, arrays, normalizer


def stage_prepare(cfg, paths):
    if (paths["output"] / "protocol.json").exists():
        print("Existing frozen design: checking hashes, no resampling.", flush=True)
        return load_context(cfg, paths)
    if __package__:
        from .initial_conditions import build_design
    else:
        from initial_conditions import build_design
    output = paths["output"]
    require(not output.exists() or not any(output.iterdir()), "Nonempty unregistered output directory; refusing to overwrite")
    identity = protocol_identity(cfg, paths)
    normalizer = read_json(paths["checkpoint"].parent / "normalizer.json")
    manifest, arrays = build_design(cfg, paths["project"], normalizer)
    output.mkdir(parents=True, exist_ok=True)
    # Persist all 64 inputs and all thresholds before constructing a model or running DG.
    write_json(output / "config_frozen.json", cfg)
    write_json(output / "cases_manifest.json", manifest)
    write_npz(output / "initial_fields.npz", **arrays)
    write_json(output / "protocol.json", {"identity": identity, "checkpoint": cfg["frozen_model"],
        "physics": cfg["physics"], "acceptance": cfg["acceptance"], "design_seed": cfg["design"]["seed"],
        "scope": "2x2 paired 64-case screening; thresholds chosen before prediction; no training",
        "reference_quality": "sampled spatial/time refinement self-consistency, not an error bound",
        "timing": cfg["benchmark"]})
    track(output, ["config_frozen.json", "cases_manifest.json", "initial_fields.npz", "protocol.json"])
    print("Prepared 64 planned cases; invalid cases remain in the manifest.", flush=True)
    return manifest, arrays, normalizer


def read_checks(output):
    path = Path(output) / "reference_checks.json"
    if path.exists():
        return read_json(verify_artifact(output, "reference_checks.json"))
    return {"interface": {}, "cases": {}, "groups": {g: {"status": "INCOMPLETE", "checked_case_ids": []} for g in GROUPS},
            "scope": "Sampled numerical self-consistency; not an exact solution or rigorous bound"}


def save_checks(output, checks):
    write_json(Path(output) / "reference_checks.json", checks)
    track(output, ["reference_checks.json"])


def new_adapter(cfg, paths):
    if __package__:
        from .dg_adapter import DGAdapter
    else:
        from dg_adapter import DGAdapter
    adapter = DGAdapter(cfg, paths["project"], paths["output"] / "dg")
    if cfg["dg"]["build_if_missing"]:
        info = adapter.build()
        write_json(paths["output"] / "build_info.json", info)
        track(paths["output"], ["build_info.json"])
    return adapter


def stage_reference(cfg, paths, manifest, arrays, normalizer):
    import numpy as np
    if __package__:
        from .metrics_report import refinement_metrics
    else:
        from metrics_report import refinement_metrics
    output = paths["output"]
    adapter = new_adapter(cfg, paths)
    checks = read_checks(output)
    try:
        checks["interface"]["dg_regression"] = adapter.run_regression(manifest["regression_sample_ids"])
    except Exception as exc:
        checks["interface"]["dg_regression"] = {"status": "FAIL", "error": str(exc)}
    baseline = np.full((64, 3, 65), np.nan)
    reference = np.full_like(baseline, np.nan)
    weights = np.asarray(normalizer["quadrature_weights"])
    scales = np.asarray(normalizer["output_scale"])
    refined_ids = set(manifest["refinement_case_ids"])
    interface_ok = checks["interface"]["dg_regression"].get("status") == "PASS"
    for i, case in enumerate(manifest["cases"]):
        case_id = case["case_id"]
        record = {"status": "INCOMPLETE", "extra_refinement_selected": case_id in refined_ids}
        checks["cases"][case_id] = record
        if case["status"] != "VALID" or not interface_ok:
            record.update(status="INPUT_INVALID" if case["status"] != "VALID" else "INTERFACE_UNCERTAIN")
            continue
        solutions = {}
        for name in ["baseline", "reference"] + (["space_refinement", "time_refinement"] if case_id in refined_ids else []):
            try:
                solved = adapter.solve(case, cfg["dg"][name], output / "dg" / "cases" / case_id / name,
                                       output / "dg" / "cq_cache" / name)
                initial_digest = hashlib.sha256(
                    np.ascontiguousarray(arrays["initial_fields"][i], dtype="<f8").tobytes()).hexdigest()
                record[name] = {"status": "AVAILABLE", "metadata": solved["metadata"],
                                "output_dir": str(solved["output_dir"]), "reused_result": solved.get("reused_result", False),
                                "projection": solved["projection"], "frozen_initial_65_sha256": initial_digest}
                if solved["projection"].get("continuous_65_sha256") != initial_digest:
                    record[name].update(status="INTERFACE_UNCERTAIN",
                        error="DG continuous initializer and frozen FNO initial samples differ; reference not accepted")
                    continue
                record[name]["continuous_initial_matches_fno"] = True
                solutions[name] = solved
                if name == "baseline":
                    baseline[i] = solved["fields"]
                elif name == "reference":
                    reference[i] = solved["fields"]
            except Exception as exc:
                record[name] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        def compare(left, right):
            return refinement_metrics(solutions[left]["fields"], solutions[right]["fields"], weights, scales,
                                      cfg["physics"]["L"], cfg["acceptance"]["refinement_abs_factor"],
                                      cfg["acceptance"]["refinement_rel_factor"])
        if "baseline" in solutions and "reference" in solutions:
            record["baseline_vs_reference"] = compare("baseline", "reference")
            record["baseline_vs_reference"]["interpretation"] = "Baseline discrepancy, not a criterion proving reference accuracy"
        if any(record.get(name, {}).get("status") == "INTERFACE_UNCERTAIN"
               for name in ("baseline", "reference", "space_refinement", "time_refinement")):
            record["status"] = "INTERFACE_UNCERTAIN"
        elif "reference" not in solutions or "baseline" not in solutions:
            record["status"] = "SOLVE_FAILED"
        elif case_id in refined_ids:
            for name, key in [("space_refinement", "space"), ("time_refinement", "time")]:
                if name in solutions:
                    record[key] = compare("reference", name)
            record["status"] = "CHECKED_PASS" if all(record.get(k, {}).get("all_pass", False) for k in ("space", "time")) else "REFERENCE_UNCERTAIN"
        else:
            record["status"] = "AVAILABLE_NOT_INDIVIDUALLY_REFINED"
        # Save every completed case, including failures; no replacement or unbounded refinement.
        save_checks(output, checks)
        print(f"Reference {i+1}/64 {case_id}: {record['status']}", flush=True)
    for group in GROUPS:
        ids = [c["case_id"] for c in manifest["cases"] if c["group"] == group]
        selected = [cid for cid in ids if cid in refined_ids]
        statuses = [checks["cases"][cid]["status"] for cid in ids]
        if not interface_ok or any(s in ("REFERENCE_UNCERTAIN", "INTERFACE_UNCERTAIN", "INPUT_INVALID") for s in statuses):
            status = "REFERENCE_UNCERTAIN"
        elif any(s in ("INCOMPLETE", "SOLVE_FAILED") for s in statuses) or len(selected) != 2:
            status = "INCOMPLETE"
        else:
            status = "SAMPLED_PASS" if all(checks["cases"][cid]["status"] == "CHECKED_PASS" for cid in selected) else "REFERENCE_UNCERTAIN"
        checks["groups"][group] = {"status": status, "checked_case_ids": selected,
            "note": "Two preselected representatives checked separately in space and time; other cases not individually certified"}
    write_npz(output / "dg_reference.npz", references=reference, baselines=baseline, x=arrays["x"], case_id=arrays["case_id"])
    track(output, ["dg_reference.npz"])
    save_checks(output, checks)
    return adapter


def ensure_predictor(cfg, paths, checks, predictor=None):
    if __package__:
        from .frozen_predictor import FrozenFNOPredictor
    else:
        from frozen_predictor import FrozenFNOPredictor
    if predictor is None:
        predictor = FrozenFNOPredictor(cfg, paths)
        checks["interface"]["fno_regression"] = predictor.regression(cfg, paths)
        save_checks(paths["output"], checks)
    require(checks["interface"].get("fno_regression", {}).get("status") == "PASS", "Frozen A regression failed; acceptance prediction stopped")
    return predictor


def stage_evaluate(cfg, paths, manifest, arrays, predictor=None):
    import numpy as np
    output = paths["output"]
    if (output / "fno_predictions.npz").exists():
        verify_artifact(output, "fno_predictions.npz")
        verify_artifact(output, "prediction_status.json")
        print("Reusing this acceptance's hashed predictions; no repeat inference.", flush=True)
        return predictor
    checks = read_checks(output)
    predictions = np.full((64, 3, 65), np.nan)
    status = {}
    try:
        predictor = ensure_predictor(cfg, paths, checks, predictor)
    except Exception as exc:
        checks["interface"]["fno_regression"] = {"status": "FAIL", "error": str(exc)}
        save_checks(output, checks)
        for case in manifest["cases"]:
            status[case["case_id"]] = {"status": "INTERFACE_UNCERTAIN", "error": str(exc)}
    else:
        for i, case in enumerate(manifest["cases"]):
            if case["status"] != "VALID":
                status[case["case_id"]] = {"status": "INPUT_INVALID"}
                continue
            try:
                fields = arrays["initial_fields"][i]
                predictions[i] = predictor.predict(case["alpha"], case["tau"], fields[0], fields[1], arrays["x"])
                status[case["case_id"]] = {"status": "AVAILABLE"}
            except Exception as exc:
                status[case["case_id"]] = {"status": "PREDICTION_FAILED", "error": str(exc)}
        write_json(output / "frozen_model_loaded.json", predictor.metadata)
        track(output, ["frozen_model_loaded.json"])
    write_npz(output / "fno_predictions.npz", predictions=predictions, x=arrays["x"], case_id=arrays["case_id"])
    write_json(output / "prediction_status.json", status)
    track(output, ["fno_predictions.npz", "prediction_status.json"])
    return predictor


def stage_report(cfg, paths, manifest, arrays, normalizer):
    import numpy as np
    if __package__:
        from .metrics_report import write_report
    else:
        from metrics_report import write_report
    output = paths["output"]
    data = dict(arrays)
    for filename, key in [("dg_reference.npz", "references"), ("fno_predictions.npz", "predictions")]:
        if (output / filename).exists():
            with np.load(verify_artifact(output, filename), allow_pickle=False) as saved:
                require(np.array_equal(saved["case_id"], arrays["case_id"]), "Mismatched artifact case IDs")
                data[key] = saved[key].copy()
        else:
            data[key] = np.full((64, 3, 65), np.nan)
    if (output / "prediction_status.json").exists():
        data["prediction_status"] = read_json(verify_artifact(output, "prediction_status.json"))
    benchmark = read_json(verify_artifact(output, "benchmark.json")) if (output / "benchmark.json").exists() else {"status": "NOT_RUN", "groups": {}}
    write_report(output, manifest, data, read_checks(output), benchmark, normalizer, cfg)
    names = ["predictions_and_reference.npz", "per_case_metrics.csv", "summary.json", "REPORT_验收结论.md"]
    names += [str(p.relative_to(output)) for p in (output / "plots").glob("*.png")] if (output / "plots").exists() else []
    track(output, [n for n in names if (output / n).is_file()])
    print(f"Report saved: {output / 'REPORT_验收结论.md'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--stage", choices=("prepare", "reference", "evaluate", "benchmark", "report", "all"), default="all")
    args = parser.parse_args(argv)
    cfg, paths = load_configuration(args.config)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    verify_sources(cfg, paths)
    context = stage_prepare(cfg, paths) if args.stage in ("prepare", "all") else load_context(cfg, paths)
    manifest, arrays, normalizer = context
    predictor, adapter = None, None
    if args.stage in ("reference", "all"):
        adapter = stage_reference(cfg, paths, manifest, arrays, normalizer)
    if args.stage in ("evaluate", "all"):
        predictor = stage_evaluate(cfg, paths, manifest, arrays, predictor)
    if args.stage in ("benchmark", "all"):
        if __package__:
            from .benchmark_acceptance import benchmark_saved_design
        else:
            from benchmark_acceptance import benchmark_saved_design
        checks = read_checks(paths["output"])
        # Integrity failures are fatal: never replace an altered record with a
        # benchmark failure summary before the user can inspect the old file.
        saved = paths["output"] / "benchmark.json"
        existing = read_json(verify_artifact(paths["output"], "benchmark.json")) if saved.exists() else {}
        reuse = cfg["benchmark"]["reuse_saved_benchmark"] and existing.get("status") == "COMPLETE"
        if reuse:
            verify_artifact(paths["output"], "latency_samples.csv")
        try:
            if reuse:
                print("Reusing verified complete timing records; no model reload or timing.", flush=True)
            else:
                predictor = ensure_predictor(cfg, paths, checks, predictor)
                adapter = adapter or new_adapter(cfg, paths)
                benchmark_saved_design(cfg, paths, manifest, arrays, predictor, adapter)
        except Exception as exc:
            write_json(paths["output"] / "benchmark.json", {"status": "FAILED", "error": str(exc), "groups": {}})
            track(paths["output"], ["benchmark.json"])
            print(f"Benchmark incomplete: {exc}", flush=True)
    if args.stage in ("report", "all"):
        stage_report(cfg, paths, manifest, arrays, normalizer)
    verify_sources(cfg, paths)


if __name__ == "__main__":
    main()
