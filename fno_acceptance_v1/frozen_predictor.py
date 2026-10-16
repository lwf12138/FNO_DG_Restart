"""Resident frozen A model. No data, model, timer or prediction runs on import."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import time


class FrozenFNOPredictor:
    def __init__(self, cfg, paths):
        import numpy as np
        import torch
        if __package__:
            from .acceptance_common import require, sha256, read_json
        else:
            from acceptance_common import require, sha256, read_json
        self._np, self._torch, self._require = np, torch, require
        checkpoint_path, fno_dir = Path(paths["checkpoint"]), Path(paths["fno"])
        # Integrity checks and dependency import time are deliberately outside cold model initialization.
        require(sha256(checkpoint_path) == cfg["frozen_model"]["sha256"], "Wrong frozen A model")
        require(cfg["frozen_model"]["device"] == "cpu", "Acceptance v1 supports explicit CPU only; no fallback")
        sys.path.insert(0, str(fno_dir))
        modules = {name: importlib.import_module(name) for name in ("model", "data", "utils")}
        for name, module in modules.items():
            require(Path(module.__file__).resolve() == (fno_dir / (name + ".py")).resolve(), "Another project's module is already imported; use a fresh Python process")
        self._encode, self._decode = modules["data"].encode_inputs, modules["data"].decode_outputs
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        started = time.perf_counter()
        payload = modules["utils"].load_checkpoint(checkpoint_path)
        for key in ("epoch", "fine_tune_step", "source_epoch"):
            require(payload[key] == cfg["frozen_model"][key], f"Wrong checkpoint {key}")
        require(payload.get("objective") == "mse", "Expected arm A, not arm B")
        self.normalizer = payload["normalizer"]
        modules["data"].validate_normalizer(self.normalizer)
        require(payload["protocol"] == modules["data"].protocol_from_normalizer(self.normalizer), "Checkpoint protocol mismatch")
        require(self.normalizer == read_json(checkpoint_path.parent / "normalizer.json"), "A normalizer sidecar mismatch")
        self.model = modules["model"].build_model(payload["model_config"])
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(torch.device("cpu"))
        self.model.eval()
        self.model.requires_grad_(False)
        self.initialization_seconds = time.perf_counter() - started
        self.first_prediction_seconds = None
        self.x = np.asarray(self.normalizer["x"], dtype=np.float64)
        self.metadata = {"checkpoint_sha256": cfg["frozen_model"]["sha256"],
            "epoch": payload["epoch"], "fine_tune_step": payload["fine_tune_step"], "source_epoch": payload["source_epoch"],
            "model_config": payload["model_config"], "environment": modules["utils"].environment_info("cpu"),
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "dtype": "float32", "device": "cpu", "answer_cache": False,
            "cold_scope": "checkpoint read + strict original model construction/load + first end-to-end call; dependency imports/hash checks/process startup excluded; OS file cache uncontrolled"}

    def predict(self, alpha, tau, E0, H0, x):
        np, torch = self._np, self._torch
        first = self.first_prediction_seconds is None
        started = time.perf_counter() if first else None
        electric, magnetic, grid = np.asarray(E0), np.asarray(H0), np.asarray(x)
        self._require(grid.ndim == 1 and electric.shape == magnetic.shape == grid.shape, "Single-call interface requires E0/H0/x vectors")
        self._require(np.asarray(alpha).ndim == np.asarray(tau).ndim == 0, "Single scalar alpha/tau required")
        encoded = self._encode(alpha, tau, electric, magnetic, grid, self.normalizer)
        self._require(encoded.shape == (1, 5, 65), "Frozen full-grid batch=1 interface only")
        with torch.inference_mode():
            result = self.model(torch.from_numpy(np.ascontiguousarray(encoded))).cpu().numpy()
        physical = self._decode(result, self.normalizer)[0]
        self._require(physical.shape == (3, 65) and np.isfinite(physical).all(), "Invalid frozen prediction")
        if first:
            self.first_prediction_seconds = time.perf_counter() - started
        return physical

    def cold_start_record(self):
        return {"initialization_seconds": self.initialization_seconds, "first_prediction_seconds": self.first_prediction_seconds,
                "combined_seconds": None if self.first_prediction_seconds is None else self.initialization_seconds + self.first_prediction_seconds,
                "scope": self.metadata["cold_scope"], "first_query": "first fixed A-validation regression sample; same T=1/65-point task"}

    def regression(self, cfg, paths):
        """Check four fixed saved A validation inputs; this is not new-function acceptance."""
        np = self._np
        folder = Path(paths["checkpoint"]).parent
        with np.load(folder / "best_validation_predictions.npz", allow_pickle=False) as data:
            saved = {name: data[name].copy() for name in ("sample_id", "prediction_physical", "x")}
        with np.load(Path(paths["project"]) / "data/dataset.npz", allow_pickle=False) as data:
            old = {name: data[name].copy() for name in ("sample_id", "initial_fields", "alpha", "tau", "x")}
        by_id = {str(value): i for i, value in enumerate(old["sample_id"])}
        n = cfg["frozen_model"]["regression_count"]
        rows = []
        scale = np.asarray(self.normalizer["output_scale"])[:, None]
        for j in range(n):
            sample_id = str(saved["sample_id"][j]); i = by_id[sample_id]
            pred = self.predict(old["alpha"][i], old["tau"][i], *old["initial_fields"][i, :2], old["x"])
            difference = np.abs(pred - saved["prediction_physical"][j])
            tolerance = cfg["frozen_model"]["regression_atol_scale_factor"] * scale + cfg["frozen_model"]["regression_rtol"] * np.abs(saved["prediction_physical"][j])
            rows.append({"sample_id": sample_id, "max_absolute_difference": float(difference.max()), "pass": bool(np.all(difference <= tolerance))})
        return {"status": "PASS" if all(r["pass"] for r in rows) else "FAIL", "samples": rows,
                "scope": "frozen A loading/encoding/decoding regression only; these are not any of the 64 new acceptance cases",
                "checkpoint": self.metadata, "cold_start": self.cold_start_record()}
