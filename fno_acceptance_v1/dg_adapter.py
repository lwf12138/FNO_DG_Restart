"""Isolated, manual-run DG adapter; importing this module never builds or solves.

Only the continuous-initial-data interface changes. The copied DG/RK-CQ kernel
keeps its spatial quadrature, PEC numerical flux and complete history unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np

if __package__:
    from .initial_conditions import evaluate_initial
else:
    from initial_conditions import evaluate_initial


HERE = Path(__file__).resolve().parent


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _within_acceptance(path):
    path = Path(path).resolve()
    if path == HERE or not path.is_relative_to(HERE):
        raise ValueError(f"DG outputs/build/cache must be below the new acceptance directory: {path}")
    return path


def monic_basis(xi, k):
    """Same recurrence as cole_cole.cpp::legendre_basis, not ordinary Legendre Pn."""
    xi = np.asarray(xi, dtype=np.float64)
    values = np.ones(xi.shape + (k + 1,), dtype=np.float64)
    if k >= 1:
        values[..., 1] = xi
    for n in range(1, k):
        values[..., n + 1] = xi * values[..., n] - n * n / (4 * n * n - 1) * values[..., n - 1]
    return values


def monic_mass(k):
    """mm_m = integral_{-1}^1 phi_m^2 / 2; physical cell mass is h_j*mm_m."""
    endpoint = monic_basis(np.array(1.0), k)
    return endpoint ** 2 / (2 * np.arange(k + 1) + 1)


def _support_boundaries(spec, L):
    boundaries = []
    if spec["family"] != "modal":
        for field in ("E", "H"):
            for pulse in spec[field]["components"]:
                c, w = float(pulse["center"]), float(pulse["half_width"])
                if not np.isfinite(c + w) or w <= 0 or not (0 < c - w < c + w < 1):
                    raise ValueError("pulse support must lie strictly inside (0,L)")
                boundaries.extend(((c - w) * L, (c + w) * L))
    return np.unique(boundaries)


def _checked_initial(spec, x, L):
    value = np.asarray(evaluate_initial(spec, np.asarray(x), L))
    if np.iscomplexobj(value) or value.shape != (3,) + np.shape(x) or not np.all(np.isfinite(value)):
        raise ValueError("continuous initial evaluator returned invalid shape/numbers")
    value = value.astype(np.float64, copy=False)
    if np.any(value[2] != 0):
        raise ValueError("acceptance protocol fixes P0=0")
    return value


def _project_once(spec, N, k, L, order, subcells):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    h = L / N
    edges = np.arange(N + 1, dtype=np.float64) * h  # Same mesh construction as C++.
    mass = monic_mass(k)
    support = _support_boundaries(spec, L)
    coefficients = np.zeros((3, N, k + 1), dtype=np.float64)
    digest = hashlib.sha256()
    evaluations = 0
    for cell in range(N):
        left, right = edges[cell], edges[cell + 1]
        J, center = (right - left) / 2, (right + left) / 2
        cuts = np.unique(np.concatenate((np.linspace(left, right, subcells + 1),
                                         support[(support > left) & (support < right)])))
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            jacobian = (hi - lo) / 2
            xq = (hi + lo) / 2 + jacobian * nodes
            values = _checked_initial(spec, xq, L)
            basis = monic_basis((xq - center) / J, k)
            coefficients[:, cell] += np.einsum("fq,q,qm->fm", values, weights * jacobian, basis)
            digest.update(xq.astype("<f8").tobytes())
            digest.update(values.astype("<f8").tobytes())
            evaluations += len(xq)
        coefficients[:, cell] /= ((right - left) * mass)[None, :]
    return coefficients, {"order": order, "subcells_per_cell": subcells,
                          "support_edges_split": support.tolist(), "quadrature_evaluations": evaluations,
                          "quadrature_nodes_and_values_sha256": digest.hexdigest()}


def project_initial(spec, settings, options=None):
    """Project the actual continuous function, compare independent higher quadrature.

    Both orders split every pulse support edge, so tiny unsampled boundary pieces
    cannot silently disappear. These are initial-projection checks, not PDE bounds.
    The higher-order coefficients are supplied to the DG kernel.
    """
    options = options or {}
    N, k, L = int(settings["N"]), int(settings["k"]), float(settings.get("L", 2 * np.pi))
    order = int(options.get("projection_order", 32))
    check_order = int(options.get("projection_check_order", 64))
    subcells = int(options.get("projection_subcells", 2))
    if N < 1 or k != 2 or order < 24 or check_order <= order or subcells < 1:
        raise ValueError("projection requires k=2, >=24 quadrature points and a higher independent check")
    low, low_info = _project_once(spec, N, k, L, order, subcells)
    high, high_info = _project_once(spec, N, k, L, check_order, subcells)
    mass = (L / N) * monic_mass(k)
    difference = np.sqrt(np.sum((low - high) ** 2 * mass[None, None, :], axis=(1, 2)))
    norm = np.sqrt(np.sum(high ** 2 * mass[None, None, :], axis=(1, 2)))
    tolerance = float(options.get("projection_atol", 1e-12)) + float(options.get("projection_rtol", 1e-10)) * norm
    passed = np.isfinite(difference).all() and bool(np.all(difference <= tolerance))
    x65 = np.linspace(0, L, int(settings["points"]))
    direct = _checked_initial(spec, x65, L)
    flattened = _checked_initial(spec, x65.reshape(5, 13), L).reshape(3, 65)
    shape_consistent = bool(np.array_equal(direct, flattened))
    info = {"basis": "monic Legendre", "physical_mass": "h_j * phi_m(1)^2/(2m+1)",
            "layout": "[field E/H/P, cell, mode] C-order little-endian float64",
            "selected_projection": "higher quadrature order", "low_quadrature": low_info,
            "high_quadrature": high_info, "projection_difference_l2": difference.tolist(),
            "projected_norm_l2": norm.tolist(), "projection_tolerance": tolerance.tolist(),
            "projection_pass": bool(passed and shape_consistent),
            "continuous_65_flat_vs_shaped_exact": shape_consistent,
            "continuous_65_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
            "continuous_function_source_sha256": _sha(HERE / "initial_conditions.py"),
            "coefficient_sha256": hashlib.sha256(high.astype("<f8").tobytes()).hexdigest(),
            "notice": "same evaluator used at network nodes and DG quadrature; projection changes representation, not continuous input"}
    return high, info


class DGAdapter:
    def __init__(self, cfg, project_root, workspace):
        self.cfg = dict(cfg.get("dg", cfg))
        self.project_root = Path(project_root).resolve()
        self.workspace = _within_acceptance(workspace)
        self.source = HERE / "reference_solver"
        self.build_dir = _within_acceptance(HERE / "reference_solver" / "build")
        self.threads = int(self.cfg.get("threads", 1))
        if self.threads != 1:
            raise ValueError("this acceptance DG build is serial: configure threads=1")

    def _sources(self):
        return {str(p.relative_to(self.source)): _sha(p) for p in sorted(self.source.rglob("*"))
                if p.is_file() and "build" not in p.relative_to(self.source).parts
                and p.suffix in {".hpp", ".cpp", ".txt"}}

    def _check_original_snapshot(self):
        manifest = json.loads((self.source / "source_manifest.json").read_text())
        for relative, digest in manifest["original_project_files_sha256"].items():
            if _sha(self.source / "original" / relative) != digest:
                raise RuntimeError(f"isolated original source snapshot changed: {relative}")
            if _sha(self.project_root / relative) != digest:
                raise RuntimeError(f"original solver source no longer matches frozen snapshot: {relative}")

    def _env(self):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", COLECOLE_DISABLE_OPENMP="1")
        return env

    def build(self):
        """Explicit manual-run action. Never called on import or construction."""
        self._check_original_snapshot()
        sources = self._sources()
        receipt_path = self.build_dir / "build_receipt.json"
        binaries = [self.build_dir / "dg_acceptance", self.build_dir / "dg_original_regression"]
        if receipt_path.exists() and all(p.is_file() for p in binaries):
            saved = json.loads(receipt_path.read_text())
            if saved["source_sha256"] == sources and all(_sha(p) == saved["binary_sha256"][p.name] for p in binaries):
                return {**saved, "reused_build": True, "build_seconds_this_call": None}
        self.build_dir.mkdir(parents=True, exist_ok=True)
        commands = [["cmake", "-S", str(self.source), "-B", str(self.build_dir), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(self.build_dir), "--parallel", "1"]]
        log_path = self.build_dir / ("build_" + uuid.uuid4().hex + ".log")
        start = time.perf_counter()
        try:
            with log_path.open("w") as log:
                for command in commands:
                    subprocess.run(command, cwd=HERE, env=self._env(), stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"isolated DG build failed; manually provide CMake/C++17/Boost, see {log_path}") from exc
        receipt = {"commands": commands, "source_sha256": sources,
                   "binary_sha256": {p.name: _sha(p) for p in binaries},
                   "build_seconds": time.perf_counter() - start,
                   "reused_build": False, "log": str(log_path), "threads": 1}
        _write_json(receipt_path, receipt)
        return receipt

    def _binary(self, name):
        path = self.build_dir / name
        receipt_path = self.build_dir / "build_receipt.json"
        if not path.is_file() or not receipt_path.is_file():
            raise RuntimeError("isolated DG binary missing; enable dg.build_if_missing and manually run acceptance")
        receipt = json.loads(receipt_path.read_text())
        if receipt["source_sha256"] != self._sources() or _sha(path) != receipt["binary_sha256"][name]:
            raise RuntimeError("isolated build source/binary hash mismatch; manually rebuild via acceptance entry")
        return path

    @staticmethod
    def _settings(settings):
        out = {"N": int(settings["N"]), "k": int(settings.get("k", 2)),
               "dt": float(settings["dt"]), "T": float(settings.get("T", 1)),
               "points": int(settings.get("points", 65)), "L": float(settings.get("L", 2 * np.pi))}
        nt = out["T"] / out["dt"] if out["dt"] > 0 else np.nan
        if (out["N"] < 1 or out["k"] != 2 or out["T"] != 1 or out["points"] != 65
                or out["L"] != 2 * np.pi or not np.isfinite(nt) or abs(nt - round(nt)) > 1e-9):
            raise ValueError("DG acceptance fixes k=2,T=1,L=2pi,65 points and integer T/dt")
        return out

    def _cache_verify(self, cache_dir):
        cache_dir = _within_acceptance(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        receipt = cache_dir / "weights_hashes.json"
        actual = {p.name: _sha(p) for p in sorted(cache_dir.glob("rk_weights_*.csv"))}
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["files"] != actual or saved["kernel_sha256"] != _sha(self.source / "src/cole_cole.cpp"):
                raise RuntimeError(f"CQ weights cache hash mismatch: {cache_dir}")
        elif actual:
            raise RuntimeError(f"untracked CQ cache files; choose a new isolated cache directory: {cache_dir}")
        return cache_dir

    def _cache_record(self, cache_dir):
        _write_json(cache_dir / "weights_hashes.json",
                    {"kernel_sha256": _sha(self.source / "src/cole_cole.cpp"),
                     "files": {p.name: _sha(p) for p in sorted(cache_dir.glob("rk_weights_*.csv"))},
                     "scope": "RK-CQ weights only; alpha/Nt/RK/CQ determine key; LU is rebuilt"})

    @staticmethod
    def _load_fields(output_dir, settings, case):
        metadata = json.loads((output_dir / "metadata.json").read_text())
        rows = np.loadtxt(output_dir / "fields.csv", delimiter=",", skiprows=1, dtype=np.float64)
        x = np.linspace(0, settings["L"], settings["points"])
        if rows.shape != (65, 5) or not np.isfinite(rows).all():
            raise RuntimeError("invalid/nonfinite DG endpoint output")
        if not np.allclose(rows[:, 0], 1, rtol=0, atol=5e-14) or not np.allclose(rows[:, 1], x, rtol=0, atol=5e-14):
            raise RuntimeError("DG endpoint/grid mismatch")
        for key in ("N", "k", "T", "points"):
            if metadata[key] != settings[key]:
                raise RuntimeError(f"DG metadata mismatch: {key}")
        for key in ("alpha", "tau"):
            if metadata[key] != float(case[key]):
                raise RuntimeError(f"DG metadata mismatch: {key}")
        if abs(metadata["dt"] - settings["dt"]) > 5e-16 or metadata["Nt"] != round(1 / settings["dt"]):
            raise RuntimeError("DG actual time grid differs from declared settings")
        return rows[:, 2:].T.copy(), metadata

    def solve(self, case, settings, output_dir, cache_dir, force=False):
        """A real DG solve; force=True always allocates a new run_UUID directory.

        force=False may load a verified old result, explicitly marked reused_result
        with no timing. Timing callers MUST use force=True and inspect cache_hit.
        """
        settings = self._settings(settings)
        binary = self._binary("dg_acceptance")
        if not 0 < float(case["alpha"]) < 1 or not np.isfinite(float(case["tau"])) or float(case["tau"]) <= 0:
            raise ValueError("invalid material parameters")
        identity = {"case_id": str(case["case_id"]), "alpha": float(case["alpha"]), "tau": float(case["tau"]),
                    "initial_spec": case["initial_spec"], "settings": settings, "projection_options": self.cfg,
                    "binary_sha256": _sha(binary), "adapter_sha256": _sha(Path(__file__)),
                    "initial_evaluator_sha256": _sha(HERE / "initial_conditions.py")}
        fingerprint = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        output_dir = _within_acceptance(output_dir)
        if force:
            output_dir = output_dir / ("run_" + uuid.uuid4().hex)
        receipt_path = output_dir / "result_receipt.json"
        if receipt_path.exists() and not force:
            saved = json.loads(receipt_path.read_text())
            if saved["identity_sha256"] != fingerprint:
                raise RuntimeError("saved DG result belongs to a different case/configuration")
            for filename, digest in saved["files_sha256"].items():
                if _sha(output_dir / filename) != digest:
                    raise RuntimeError(f"saved DG output hash mismatch: {filename}")
            fields, metadata = self._load_fields(output_dir, settings, case)
            return {"fields": fields, "metadata": metadata, "wall_seconds": None,
                    "projection_seconds": None, "reused_result": True,
                    "projection": saved["projection"], "output_dir": str(output_dir)}
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"partial/existing DG output is preserved; use force=True for a new attempt: {output_dir}")
        cache_dir = self._cache_verify(cache_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "request.json", identity)
        start = time.perf_counter()
        projection_start = time.perf_counter()
        coefficients, projection = project_initial(case["initial_spec"], settings, self.cfg)
        projection_seconds = time.perf_counter() - projection_start
        if not projection["projection_pass"]:
            _write_json(output_dir / "projection_check.json", projection)
            raise RuntimeError(f"initial projection/interface check failed; reference remains uncertain: {output_dir}")
        coeff_path = output_dir / "requested_initial.f64"
        coefficients.astype("<f8").tofile(coeff_path)
        command = [str(binary), "--alpha", format(float(case["alpha"]), ".17g"),
                   "--tau", format(float(case["tau"]), ".17g"), "--initial-coefficients", str(coeff_path),
                   "--N", str(settings["N"]), "--k", str(settings["k"]), "--T", "1",
                   "--dt", format(settings["dt"], ".17g"), "--points", "65",
                   "--cache", str(cache_dir), "--output", str(output_dir)]
        process_start = time.perf_counter()
        with (output_dir / "solver.log").open("w") as log:
            completed = subprocess.run(command, cwd=HERE, env=self._env(), stdout=log, stderr=subprocess.STDOUT)
        process_seconds = time.perf_counter() - process_start
        if completed.returncode:
            self._cache_record(cache_dir)
            _write_json(output_dir / "failure.json", {"returncode": completed.returncode, "command": command,
                                                       "status": "DG_FAILED", "projection": projection})
            raise RuntimeError(f"DG solve failed; results preserved at {output_dir}")
        fields, metadata = self._load_fields(output_dir, settings, case)
        accepted = np.fromfile(output_dir / "initial_coefficients.f64", dtype="<f8").reshape(coefficients.shape)
        if not np.array_equal(accepted, coefficients):
            raise RuntimeError("C++ initial coefficient transport was not exact")
        wall_seconds = time.perf_counter() - start
        self._cache_record(cache_dir)
        projection["cpp_coefficient_transport_exact"] = True
        metadata.update(initial_projection_seconds=projection_seconds, wrapper_wall_seconds=wall_seconds,
                        process_wall_seconds=process_seconds, wrapper_timing_scope="projection (both quadratures including internal checks/checksums), coefficient IO, subprocess, output IO and validation; excludes build, external provenance/CQ cache hash bookkeeping and post-run report",
                        wrapper_cache_hash_seconds_included=False)
        _write_json(output_dir / "metadata.json", metadata)
        _write_json(output_dir / "projection_check.json", projection)
        _write_json(output_dir / "invocation.json", {"command": command, "cwd": str(HERE), "returncode": completed.returncode,
                                                    "process_wall_seconds": process_seconds, "wrapper_wall_seconds": wall_seconds,
                                                    "threads": 1, "cache_dir": str(cache_dir)})
        files = ["fields.csv", "initial_coefficients.f64", "final_coefficients.f64", "metadata.json", "projection_check.json", "request.json"]
        receipt = {"identity_sha256": fingerprint, "projection": projection,
                   "files_sha256": {name: _sha(output_dir / name) for name in files}}
        _write_json(receipt_path, receipt)
        return {"fields": fields, "metadata": metadata, "wall_seconds": wall_seconds,
                "projection_seconds": projection_seconds, "process_wall_seconds": process_seconds,
                "projection": projection, "reused_result": False, "output_dir": str(output_dir)}

    def run_regression(self, sample_ids=("sample_0000", "sample_0001")):
        """Manual-only old/new interface check using existing modal data, no FNO.

        The old target uses unmodified copied driver/kernel and outputs 0,T solely
        for the regression. Its timing is not used as the endpoint benchmark.
        """
        if len(sample_ids) < 2 or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("regression requires at least two different existing sample IDs")
        old_binary = self._binary("dg_original_regression")
        settings = self._settings({"N": 64, "k": 2, "dt": .0005, "T": 1, "points": 65})
        root = _within_acceptance(self.workspace / "interface_regression" / ("run_" + uuid.uuid4().hex))
        root.mkdir(parents=True)
        results = []
        dataset = self.project_root / "data/dataset.npz"
        with np.load(dataset, allow_pickle=False) as data:
            ids = data["sample_id"].astype(str)
            for sample_id in sample_ids:
                indices = np.flatnonzero(ids == sample_id)
                if len(indices) != 1:
                    raise ValueError(f"regression sample not unique in original data: {sample_id}")
                i = int(indices[0]); a, b = data["initial_coefficients"][i]
                case = {"case_id": str(sample_id), "alpha": float(data["alpha"][i]), "tau": float(data["tau"][i]),
                        "initial_spec": {"family": "modal", "a": a.tolist(), "b": b.tolist()}}
                old_dir = root / sample_id / "old"; old_dir.mkdir(parents=True)
                fmt = lambda v: ",".join(format(float(number), ".17g") for number in v)
                cache = self._cache_verify(root / "cq_cache")
                command = [str(old_binary), "--alpha", format(case["alpha"], ".17g"), "--tau", format(case["tau"], ".17g"),
                           "--a", fmt(a), "--b", fmt(b), "--N", "64", "--k", "2", "--dt", "0.0005", "--T", "1",
                           "--times", "0,1", "--points", "65", "--cache", str(cache), "--output", str(old_dir)]
                with (old_dir / "solver.log").open("w") as log:
                    completed = subprocess.run(command, cwd=HERE, env=self._env(), stdout=log, stderr=subprocess.STDOUT)
                self._cache_record(cache)
                if completed.returncode:
                    results.append({"sample_id": sample_id, "status": "FAIL", "reason": "old copied DG process failed"})
                    continue
                try:
                    new = self.solve(case, settings, root / sample_id / "new", cache)
                    old_coeff = np.fromfile(old_dir / "coefficients.f64", dtype="<f8").reshape(2, 3, 64, 3)[0]
                    new_coeff = np.fromfile(Path(new["output_dir"]) / "initial_coefficients.f64", dtype="<f8").reshape(3, 64, 3)
                    old_rows = np.loadtxt(old_dir / "fields.csv", delimiter=",", skiprows=1).reshape(2, 65, 5)
                    old_fields = old_rows[1, :, 2:].T
                    if not np.allclose(old_rows[1, :, 0], 1, atol=5e-14, rtol=0):
                        raise RuntimeError("old regression endpoint mismatch")
                    w = np.full(65, 2 * np.pi / 64); w[[0, -1]] *= .5
                    end_abs = np.sqrt(np.sum((new["fields"] - old_fields) ** 2 * w, axis=1))
                    old_norm = np.sqrt(np.sum(old_fields ** 2 * w, axis=1))
                    coef_abs = np.max(np.abs(new_coeff - old_coeff), axis=(1, 2))
                    coef_limit = 1e-12 + 1e-10 * np.max(np.abs(old_coeff), axis=(1, 2))
                    endpoint_limit = 1e-11 + 1e-9 * old_norm
                    continuous = _checked_initial(case["initial_spec"], data["x"], 2 * np.pi)
                    initial_65_max_diff = np.max(np.abs(continuous - data["initial_fields"][i]), axis=1)
                    passed = bool(np.all(coef_abs <= coef_limit) and np.all(end_abs <= endpoint_limit)
                                  and np.all(initial_65_max_diff <= 1e-13) and new["projection"]["projection_pass"])
                    results.append({"sample_id": sample_id, "status": "PASS" if passed else "FAIL",
                                    "initial_coefficient_max_abs_difference": coef_abs.tolist(),
                                    "initial_coefficient_tolerance": coef_limit.tolist(), "endpoint_absolute_l2": end_abs.tolist(),
                                    "endpoint_tolerance": endpoint_limit.tolist(),
                                    "continuous_65_vs_original_initial_max_abs": initial_65_max_diff.tolist(),
                                    "new_output_dir": new["output_dir"], "old_output_dir": str(old_dir)})
                except (RuntimeError, ValueError, OSError) as exc:
                    results.append({"sample_id": sample_id, "status": "FAIL", "reason": str(exc)})
        report = {"status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL",
                  "sample_count": len(results), "samples": results, "dataset_sha256": _sha(dataset),
                  "scope": "same DG kernel/config; original sine/cosine projection vs full-coefficient interface",
                  "output_dir": str(root), "not_a_reference_error_bound": True}
        _write_json(root / "regression.json", report)
        return report
