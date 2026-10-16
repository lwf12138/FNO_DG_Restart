"""User-triggered timers for resident FNO and actual DG solves; no import work."""
from __future__ import annotations

import csv
from pathlib import Path
import time
import uuid

if __package__:
    from .acceptance_common import read_json, track, verify_artifact, write_json
else:
    from acceptance_common import read_json, track, verify_artifact, write_json


def stats(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        return {"count": len(values), "median_seconds": None, "p95_seconds": None, "max_seconds": None}
    return {"count": len(values), "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.quantile(values, .95)), "max_seconds": float(np.max(values))}


def benchmark_saved_design(cfg, paths, manifest, arrays, predictor, adapter):
    output = paths["output"]
    if cfg["benchmark"]["reuse_saved_benchmark"] and (output / "benchmark.json").exists():
        existing = read_json(verify_artifact(output, "benchmark.json"))
        if existing.get("status") == "COMPLETE":
            verify_artifact(output, "latency_samples.csv")
            print("Using this protocol's saved timings; not claiming they were rerun.", flush=True)
            return existing
    cases = manifest["cases"]
    by_id = {c["case_id"]: (i, c) for i, c in enumerate(cases)}
    representative_ids = manifest["representative_case_ids"]
    row_fields = ["backend", "phase", "group", "case_id", "setting", "requested_cache_mode", "observed_cache_hit",
                  "repeat", "seconds", "scope", "process_wall_seconds", "solve_seconds", "projection_seconds",
                  "cache_verified", "error", "output_dir"]
    # A new isolated benchmark attempt; never clear user caches or prior attempts.
    attempt = output / "benchmark_runs" / ("attempt_" + uuid.uuid4().hex[:12])
    attempt.mkdir(parents=True, exist_ok=False)
    raw_path = attempt / "latency_samples.csv"
    rows, errors = [], []
    report = {"status": "COMPLETE", "cold_start": predictor.cold_start_record(), "groups": {},
              "frozen_model": predictor.metadata, "attempt_directory": str(attempt),
              "timing_protocol": {"fno_batch": 1, "fno_warmups": cfg["benchmark"]["fno_warmups"],
                  "fno_measured_per_group": cfg["benchmark"]["fno_repeats_per_group"],
                  "fno_scope": "resident encoding/input validation + tensor conversion + forward + NumPy transfer + decoding; no answer cache",
                  "cold_scope": predictor.metadata["cold_scope"], "threads": 1, "device": "cpu",
                  "representative_case_ids": representative_ids, "dg_repeats_per_case_per_cache_mode": 3,
                  "dg_configurations": {name: cfg["dg"][name] for name in ("baseline", "reference")},
                  "dg_wrapper_scope": "current Python wrapper including continuous initial L2 projection/checks/checksums, process start, IO, solve, sampling and result validation; external provenance/CQ cache hash bookkeeping excluded",
                  "dg_solve_scope": "C++ solve_seconds only; Python initial projection/process/IO and terminal sampling excluded",
                  "dg_cache_scope": "disk CQ weights only; factorization is rebuilt in every process; OS filesystem cache uncontrolled",
                  "fixed_material_query": "warm cache primed with another initial condition at identical alpha/tau; answers never reused",
                  "new_material_query": "fresh isolated CQ directory for every cache-miss query",
                  "speedup_scope": "median DG wrapper (or separate solve-only diagnostic) / median resident FNO, same two representatives per group",
                  "restriction": "current wrappers and two DG configurations; not proof against optimal same-accuracy DG; timing small representative set only"}}
    valid_reps = [cid for cid in representative_ids if by_id[cid][1]["status"] == "VALID"]
    if len(valid_reps) != len(representative_ids):
        errors.append("One or more preselected representatives has invalid input; no replacement was made")
    successful_warmups = 0
    with raw_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row_fields)
        writer.writeheader()
        def append(row):
            rows.append(row)
            writer.writerow(row)
            stream.flush()
        # Warm-up uses distinct planned inputs, not repeated checkpoint loading.
        for j in range(cfg["benchmark"]["fno_warmups"]):
            if not valid_reps:
                break
            i, c = by_id[valid_reps[j % len(valid_reps)]]
            warmup_row = dict(backend="FNO", phase="warmup", group=c["group"],
                              case_id=c["case_id"], setting="frozen_A", repeat=j,
                              scope="untimed warmup; excluded from measured latency")
            try:
                fields = arrays["initial_fields"][i]
                predictor.predict(c["alpha"], c["tau"], fields[0], fields[1], arrays["x"])
                successful_warmups += 1
            except Exception as exc:
                warmup_row["error"] = str(exc)
                errors.append(f"FNO warmup failed for {c['case_id']}: {exc}")
            append(warmup_row)
        # Equal batch=1 queries of the exact DG representatives; each group has >=200 calls.
        for j in range(cfg["benchmark"]["fno_repeats_per_group"]):
            for group in ("G1", "G2", "G3", "G4"):
                ids = [cid for cid in representative_ids if by_id[cid][1]["group"] == group]
                cid = ids[j % len(ids)]; i, c = by_id[cid]
                row = dict(backend="FNO", phase="measured", group=group, case_id=cid, setting="frozen_A", repeat=j,
                           scope="resident_batch1_end_to_end")
                if c["status"] != "VALID":
                    row["error"] = "INPUT_INVALID"
                    append(row)
                    continue
                try:
                    fields = arrays["initial_fields"][i]
                    started = time.perf_counter_ns()
                    predictor.predict(c["alpha"], c["tau"], fields[0], fields[1], arrays["x"])
                    elapsed = (time.perf_counter_ns() - started) * 1e-9
                    row["seconds"] = elapsed
                except Exception as exc:
                    row["error"] = str(exc)
                append(row)
        for cid in representative_ids:
            i, case = by_id[cid]
            for setting_name in cfg["benchmark"]["settings"]:
                settings = cfg["dg"][setting_name]
                warm_cache = attempt / "cq_cache" / setting_name / cid / "warm"
                # This is a real solve with another initial at the same material.
                alternatives = [c for c in cases if c["group"] == case["group"] and c["material_id"] == case["material_id"]
                                and c["case_id"] != cid and c["status"] == "VALID"]
                if case["status"] == "VALID" and alternatives:
                    try:
                        warm = adapter.solve(alternatives[0], settings, attempt / "warmup" / setting_name / cid, warm_cache, force=True)
                        append(dict(backend="DG", phase="warmup", group=case["group"], case_id=alternatives[0]["case_id"],
                                    setting=setting_name, requested_cache_mode="cache_prime", observed_cache_hit=warm["metadata"]["cache_hit"],
                                    seconds=warm["wall_seconds"], scope="wrapper", output_dir=str(warm["output_dir"])))
                    except Exception as exc:
                        errors.append(f"DG cache prime failed {cid}/{setting_name}: {exc}")
                else:
                    errors.append(f"No valid predeclared same-material warmup for {cid}/{setting_name}")
                for mode in cfg["benchmark"]["cache_modes"]:
                    for repeat in range(cfg["benchmark"]["dg_repeats_per_cache_mode"]):
                        cache = warm_cache if mode == "cache_hit" else attempt / "cq_cache" / setting_name / cid / f"cold_{repeat}"
                        row = dict(backend="DG", phase="measured", group=case["group"], case_id=cid, setting=setting_name,
                                   requested_cache_mode=mode, repeat=repeat, scope="wrapper")
                        if case["status"] != "VALID":
                            row["error"] = "INPUT_INVALID"
                            append(row)
                            continue
                        try:
                            solved = adapter.solve(case, settings, attempt / "queries" / setting_name / cid / mode, cache, force=True)
                            hit = solved["metadata"]["cache_hit"]
                            row.update(seconds=solved["wall_seconds"], solve_seconds=solved["metadata"]["solve_seconds"],
                                       process_wall_seconds=solved["metadata"].get("process_wall_seconds"),
                                       projection_seconds=solved["projection_seconds"], observed_cache_hit=hit,
                                       cache_verified=hit == (mode == "cache_hit") and not solved.get("reused_result", False),
                                       output_dir=str(solved["output_dir"]))
                            if not row["cache_verified"]:
                                row["error"] = "Observed cache state disagrees with requested timing regime"
                        except Exception as exc:
                            row["error"] = str(exc)
                        append(row)
                print(f"DG timing complete for {cid}/{setting_name}; no cached terminal answer used.", flush=True)
    warmup_complete = successful_warmups == cfg["benchmark"]["fno_warmups"]
    report["timing_protocol"]["fno_successful_warmups"] = successful_warmups
    report["timing_protocol"]["fno_warmup_status"] = "COMPLETE" if warmup_complete else "PARTIAL"
    for group in ("G1", "G2", "G3", "G4"):
        frows = [r for r in rows if r["backend"] == "FNO" and r["phase"] == "measured" and r["group"] == group]
        fno_ok = warmup_complete and len(frows) == cfg["benchmark"]["fno_repeats_per_group"] and not any(r.get("error") for r in frows)
        fstats = stats([r["seconds"] for r in frows if "seconds" in r])
        fstats["status"] = "COMPLETE" if fno_ok else "PARTIAL"
        group_result = {"fno": fstats, "dg": {}, "speedups": {},
            "timed_case_ids": [cid for cid in representative_ids if by_id[cid][1]["group"] == group],
            "actual_timing_material_count": len({by_id[cid][1]["material_id"] for cid in representative_ids if by_id[cid][1]["group"] == group})}
        for setting in cfg["benchmark"]["settings"]:
            group_result["dg"][setting] = {}
            for mode in cfg["benchmark"]["cache_modes"]:
                selected = [r for r in rows if r["backend"] == "DG" and r["phase"] == "measured" and r["group"] == group
                            and r["setting"] == setting and r["requested_cache_mode"] == mode]
                dg_ok = len(selected) == 2 * cfg["benchmark"]["dg_repeats_per_cache_mode"] and not any(r.get("error") for r in selected)
                values = {"solve": stats([r["solve_seconds"] for r in selected if "solve_seconds" in r]),
                          "wrapper": stats([r["seconds"] for r in selected if "seconds" in r]),
                          "projection": stats([r["projection_seconds"] for r in selected if "projection_seconds" in r]),
                          "status": "COMPLETE" if dg_ok else "PARTIAL"}
                group_result["dg"][setting][mode] = values
                for measure in ("wrapper", "solve"):
                    numerator, denominator = values[measure]["median_seconds"], fstats["median_seconds"]
                    group_result["speedups"][f"{setting}_{mode}_{measure}_over_fno"] = numerator / denominator if dg_ok and fno_ok and denominator and numerator is not None else None
                if not dg_ok:
                    errors.append(f"Incomplete timing regime {group}/{setting}/{mode}; no speedup assigned")
        if not fno_ok:
            errors.append(f"Incomplete FNO timing {group}; no speedup assigned")
        report["groups"][group] = group_result
    report["errors"] = errors
    report["status"] = "PARTIAL" if errors else "COMPLETE"
    report["timing_protocol"]["build_info"] = read_json(output / "build_info.json") if (output / "build_info.json").exists() else None
    write_json(attempt / "benchmark.json", report)
    # Current summary is an explicit pointer/copy of this preserved attempt, not a hidden rerun.
    (output / "latency_samples.csv").write_bytes(raw_path.read_bytes())
    write_json(output / "benchmark.json", report)
    track(output, ["latency_samples.csv", "benchmark.json"])
    return report
