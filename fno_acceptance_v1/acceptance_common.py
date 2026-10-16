"""Filesystem/provenance helpers; no work is started on import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve(path, base):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(base) / path).resolve()


def load_configuration(path=None):
    path = resolve(path or "config.json", HERE)
    cfg = read_json(path)
    project = resolve(cfg["paths"]["project_root"], path.parent)
    fno = resolve(cfg["paths"]["fno_directory"], project)
    checkpoint = resolve(cfg["paths"]["checkpoint"], fno)
    output = resolve(cfg["paths"]["output_directory"], path.parent)
    require(HERE in output.parents, "All acceptance artifacts must stay under fno_acceptance_v1")
    require(output.relative_to(HERE).parts[0] != "reference_solver",
            "Runtime outputs must not overlap the isolated solver source/build directory")
    require(cfg["schema_version"] == 1, "Unsupported acceptance schema")
    require(cfg["frozen_model"]["device"] == "cpu" and cfg["frozen_model"]["threads"] == 1
            and cfg["dg"]["threads"] == 1, "This v1 comparison is explicitly CPU single-thread only")
    require(cfg["frozen_model"]["sha256"] == "79ce920875f116020a1801b5ecc1ea4e647f5c02a3dd2e57e87de3599e8b7205",
            "Acceptance v1 freezes the specified A checkpoint, not the epoch2478 source")
    expected_physics = {"L": 6.283185307179586, "epsilon_b": 2., "mu": 1.,
        "delta_epsilon": 1., "T": 1., "P0": 0., "history_before_t0": "zero",
        "points": 65, "boundary": "PEC", "source_free": True}
    require(cfg["physics"] == expected_physics,
            "Acceptance v1 fixes the original domain, material constants, PEC, T=1, P0 and zero initial history")
    require(type(cfg["frozen_model"]["regression_count"]) is int
            and 2 <= cfg["frozen_model"]["regression_count"] <= 64,
            "Loading regression must use 2 to 64 fixed A validation samples")
    require(cfg["design"]["material_count"] == cfg["design"]["initials_per_material"] == 4,
            "Exactly four materials and four initial conditions per group")
    require(cfg["dg"]["additional_refinements"] == 0, "No automatic additional refinement is implemented")
    expected = {"baseline": (64, .0005), "reference": (128, .00025),
                "space_refinement": (256, .00025), "time_refinement": (128, .000125)}
    for name, (n, dt) in expected.items():
        require(cfg["dg"][name] == {"N": n, "k": 2, "dt": dt, "T": 1., "points": 65},
                f"Keep the predeclared separate refinement design: {name}")
    bench = cfg["benchmark"]
    require(type(bench["fno_warmups"]) is int and bench["fno_warmups"] >= 20
            and type(bench["fno_repeats_per_group"]) is int and bench["fno_repeats_per_group"] >= 200,
            "Require >=20 warmups and >=200 measured batch=1 queries per group")
    require(bench["dg_repeats_per_cache_mode"] == 3 and bench["representatives_per_group"] == 2,
            "DG timing uses two fixed representatives per group and three repeats per cache mode")
    require(bench["settings"] == ["baseline", "reference"] and bench["cache_modes"] == ["cache_hit", "cache_miss"],
            "Both DG configurations and both CQ-cache states must be reported")
    return cfg, {"config": path, "project": project, "fno": fno, "checkpoint": checkpoint, "output": output}


def verify_sources(cfg, paths):
    require(sha256(paths["checkpoint"]) == cfg["frozen_model"]["sha256"], "A checkpoint SHA256 mismatch")
    for relative, expected in cfg["source_locks"].items():
        if relative.startswith("fno_training/"):
            path = paths["fno"] / relative[len("fno_training/"):]
        else:
            path = paths["project"] / relative
        require(path.is_file() and sha256(path) == expected, f"Frozen source/data changed or missing: {relative}")
    return dict(cfg["source_locks"], checkpoint=cfg["frozen_model"]["sha256"])


def implementation_hashes():
    result = {}
    # Enumerate source roots, not arbitrary user-selected artifact directories.
    candidates = list(HERE.glob("*.py")) + list((HERE / "reference_solver").rglob("*"))
    for path in sorted(candidates):
        rel = path.relative_to(HERE)
        if any(p in {"build", "__pycache__"} for p in rel.parts):
            continue
        if path.is_file() and (path.suffix in {".py", ".cpp", ".hpp"}
                              or path.name in {"CMakeLists.txt", "source_manifest.json"}):
            result[str(rel)] = sha256(path)
    return result


def track(output, names):
    index = Path(output) / "hashes.json"
    data = read_json(index) if index.exists() else {"artifacts": {}}
    for name in names:
        data["artifacts"][str(name)] = sha256(Path(output) / name)
    write_json(index, data)


def verify_artifact(output, name):
    data = read_json(Path(output) / "hashes.json")
    require(name in data["artifacts"] and sha256(Path(output) / name) == data["artifacts"][name],
            f"Artifact is missing, changed or incomplete: {name}")
    return Path(output) / name
