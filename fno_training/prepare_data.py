#!/usr/bin/env python3
"""Explicitly prepare the fixed T=1 FNO dataset; never run a DG solver or model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .data import (INPUT_CHANNELS, METHOD, OUTPUT_CHANNELS, SPLITS, _load_raw,
                       _training_scales, encode_inputs, validate_normalizer)
    from .utils import add_config_arguments, load_config, sha256_file
else:
    from data import (INPUT_CHANNELS, METHOD, OUTPUT_CHANNELS, SPLITS, _load_raw,
                      _training_scales, encode_inputs, validate_normalizer)
    from utils import add_config_arguments, load_config, sha256_file


def prepare(cfg: dict) -> dict:
    settings = cfg["data"]
    prepared_path = Path(settings["prepared_path"])
    norm_path = Path(settings["normalizer_path"])
    sources = {Path(settings["raw_path"]).resolve(), Path(settings["raw_config_path"]).resolve()}
    if prepared_path.resolve() == norm_path.resolve() or prepared_path.resolve() in sources or norm_path.resolve() in sources:
        raise ValueError("Prepared, normalizer and source paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in (prepared_path, norm_path)):
        raise FileExistsError("Prepared outputs already exist; refusing to overwrite or normalize again")
    raw = _load_raw(cfg)
    input_scale, output_scale = _training_scales(raw)
    normalizer = dict(
        schema_version=1, method=METHOD, fitted_on="train",
        input_scale=input_scale.tolist(), output_scale=output_scale.tolist(),
        ranges=raw["ranges"], x=raw["x"].tolist(), quadrature_weights=raw["weights"].tolist(),
        target_time=raw["target_time"], input_channels=list(INPUT_CHANNELS),
        output_channels=list(OUTPUT_CHANNELS), tolerances=raw["tolerances"],
        raw_sha256=raw["raw_sha256"], split_sha256=raw["split_sha256"],
        raw_config_sha256=raw["raw_config_sha256"],
    )
    validate_normalizer(normalizer)
    X = encode_inputs(raw["alpha"], raw["tau"], raw["initial_fields"][:, 0],
                      raw["initial_fields"][:, 1], raw["x"], normalizer)
    Y = (raw["Y_physical"] / output_scale[None, :, None]).astype(np.float32)
    if not np.isfinite(Y).all():
        raise ValueError("Normalized outputs cannot be represented as finite float32")
    normalizer_text = json.dumps(normalizer, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another process preparing the same paths.
    with norm_path.open("x", encoding="utf-8") as handle:
        handle.write(normalizer_text)
    norm_hash = sha256_file(norm_path)
    with prepared_path.open("xb") as handle:
        np.savez_compressed(
            handle, X=X, Y=Y, Y_physical=raw["Y_physical"], x=raw["x"],
            quadrature_weights=raw["weights"], split=raw["split"],
            sample_id=raw["sample_id"], material_id=raw["material_id"],
            target_time=np.array(raw["target_time"]), output_scale=output_scale,
            normalizer_json=np.array(normalizer_text), normalizer_sha256=np.array(norm_hash),
        )
    return dict(prepared_path=str(prepared_path), normalizer_path=str(norm_path),
                samples=len(X), X_shape=list(X.shape), Y_shape=list(Y.shape),
                split_counts={name: int(np.sum(raw["split"] == name)) for name in SPLITS},
                raw_sha256=raw["raw_sha256"], prepared_sha256=sha256_file(prepared_path),
                normalizer_sha256=norm_hash, split_sha256=raw["split_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    print(json.dumps(prepare(cfg), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
