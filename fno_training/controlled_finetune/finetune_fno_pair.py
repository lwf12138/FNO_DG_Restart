"""Explicit user-run paired fine-tuning, never strict resume or automatic training.

Only --config is accepted. Importing this module does not load data, torch,
checkpoints, models or optimizers. A and B are separate manual invocations.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIELDS = ("E", "H", "P")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def resolved(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def shared_config(cfg):
    return {k: v for k, v in cfg.items() if k not in {"variant", "objective", "peer_config"}}


def validate_pair(config_path):
    """Read both configs and reject unpaired overrides before loading torch."""
    config_path = resolved(config_path, HERE)
    cfg = read_json(config_path)
    require(cfg["schema_version"] == 1, "Unsupported paired experiment schema")
    require(cfg["variant"] in ("A", "B"), "variant must be A or B")
    require(cfg["objective"] == {"A": "mse", "B": "mixed"}[cfg["variant"]], "Variant/objective mismatch")
    peer_path = resolved(cfg["peer_config"], config_path.parent)
    peer = read_json(peer_path)
    require(peer_path.parent == config_path.parent, "Keep the paired configs in the same directory")
    require(peer["variant"] == {"A": "B", "B": "A"}[cfg["variant"]], "Peer must be the other arm")
    require(peer["objective"] == {"A": "mse", "B": "mixed"}[peer["variant"]], "Peer objective mismatch")
    require(resolved(peer["peer_config"], peer_path.parent) == config_path, "Peer config must point back")
    require(shared_config(cfg) == shared_config(peer), "A/B settings differ beyond variant/objective; edit both together")
    t, loss, comparison = cfg["training"], cfg["loss"], cfg["comparison"]
    require(t["device"] == "cpu" and t["optimizer"] == "AdamW", "This controlled pair requires CPU AdamW")
    require(t["reset_optimizer"] is True and t["reset_scheduler"] is True, "Both states must reset")
    require(t["early_stopping"] is False and t["num_workers"] == 0, "Fixed budget requires no early stopping, workers=0")
    require(t["deterministic_algorithms"] is True, "Use deterministic algorithms or stop on unsupported operations")
    for key in ("batch_size", "max_updates", "validation_every_updates", "cpu_threads"):
        require(type(t[key]) is int and t[key] > 0, f"training.{key} must be a positive integer")
    require(type(t["seed"]) is int and t["seed"] >= 0, "seed must be a nonnegative integer")
    for key in ("learning_rate", "min_learning_rate", "eps", "weight_decay"):
        require(type(t[key]) in (int, float) and math.isfinite(t[key]) and t[key] > 0, f"Invalid {key}")
    require(t["min_learning_rate"] <= t["learning_rate"] and t["max_updates"] > 1, "Invalid cosine schedule endpoints")
    require(t["schedule"] == "cosine_by_update", "Only the declared fixed update schedule is implemented")
    require(len(t["betas"]) == 2 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v < 1 for v in t["betas"]), "Invalid AdamW betas")
    require(type(loss["lambda"]) in (int, float) and math.isfinite(loss["lambda"]) and 0 < loss["lambda"] <= 1,
            "B must have a finite positive mixture coefficient <=1")
    require(len(loss["delta"]) == 3 and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in loss["delta"]), "Invalid delta")
    require(loss["delta_space"] == "normalized_output_rms", "delta is an amplitude in saved RMS units")
    expected_comparison = {"primary": "macro_mean_relative_l2", "mode": "min", "relative_norm_threshold": 1e-10,
        "undefined_policy": "fail_if_any_validation_relative_undefined", "checkpoint_selection": "same_primary_including_step_zero",
        "absolute_review": "report_each_field_mean_p95_max_and_flag_any_increase", "joint_relative_threshold": .01, "use_test": False}
    require(comparison == expected_comparison, "This pair uses the predeclared common validation comparison protocol")
    require(cfg["source_epoch"] == 2478 and cfg["expected_counts"] == {"train": 384, "validation": 64}, "Wrong baseline/data population")
    base = config_path.parent
    project_config = resolved(cfg["project_config"], base)
    source = resolved(cfg["source_checkpoint"], base)
    require(digest(project_config) == cfg["expected_project_config_sha256"], "Original config changed; review before defining a new pair")
    require(digest(source) == cfg["source_checkpoint_sha256"], "Baseline checkpoint bytes changed")
    for name, expected in cfg["expected_original_sources"].items():
        require(digest(PROJECT / name) == expected, f"Original implementation changed: {name}")
    original = read_json(source.parent / "config.json")
    require(cfg["expected_model_config"] == original["model"], "Architecture must match the frozen baseline")
    require(t["weight_decay"] == original["training"]["weight_decay"], "Do not change baseline weight decay")
    require(t["betas"] == [.9, .999] and t["eps"] == 1e-8, "Preserve baseline AdamW beta/epsilon defaults")
    root = resolved(cfg["output_root"], base)
    require(HERE in root.parents and root != HERE, "New runs must remain in this new experiment directory")
    return cfg, config_path, peer_path, project_config, source, root


def lr_factor(completed_updates, total_updates, floor_ratio):
    """Update j=0..U-1 uses lr_min+(lr_max-lr_min)*(1+cos(pi*j/(U-1)))/2."""
    position = min(completed_updates, total_updates - 1) / (total_updates - 1)
    return floor_ratio + (1 - floor_ratio) * (1 + math.cos(math.pi * position)) / 2


def validation_record(step, metrics):
    record = {"step": step, "validation_mse": metrics["mse"], "validation_mixed_loss": metrics["mixed_loss"],
              "validation_macro_mean_relative_l2": metrics["macro_mean_relative_l2"],
              "validation_joint_relative_le_1pct": metrics["joint_relative_le_1pct"]["fraction_all"]}
    for field in FIELDS:
        for metric in ("absolute_l2", "relative_l2"):
            for stat, value in metrics["fields"][field][metric].items():
                record[f"validation_{field}_{metric}_{stat}"] = value
    return record


def run(config_path):
    cfg, config_path, peer_path, project_path, source, output_root = validate_pair(config_path)
    # Imports occur only after an explicit CLI/manual run and pair validation.
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    if __package__:
        from .paired_objectives import loss_terms, summarize_validation
    else:
        from paired_objectives import loss_terms, summarize_validation
    sys.path.insert(0, str(PROJECT))
    from data import load_prepared, decode_outputs
    from model import build_model
    from utils import (load_config, load_checkpoint, seed_all, environment_info, new_run_dir,
                       write_json, cpu_model_state, save_checkpoint)

    t, loss = cfg["training"], cfg["loss"]
    project_cfg = load_config(project_path)
    bundle = load_prepared(project_cfg)  # Validates stored data; does not regenerate or refit it.
    checkpoint = load_checkpoint(source)
    require(checkpoint["epoch"] == cfg["source_epoch"] and checkpoint.get("kind") == "best", "Expected the frozen best")
    for key, expected in (("model_config", cfg["expected_model_config"]), ("normalizer", bundle.normalizer),
                          ("protocol", bundle.protocol), ("data_fingerprints", bundle.fingerprints)):
        require(checkpoint[key] == expected, f"Baseline {key} mismatch")
    require(bundle.fingerprints == cfg["expected_data_fingerprints"], "Original dataset fingerprints changed")
    train_indices, val_indices = bundle.indices("train"), bundle.indices("validation")
    require(len(train_indices) == cfg["expected_counts"]["train"] and len(val_indices) == cfg["expected_counts"]["validation"], "Wrong split counts")
    batches_per_epoch = math.ceil(len(train_indices) / t["batch_size"])
    require(t["validation_every_updates"] == batches_per_epoch and t["max_updates"] % batches_per_epoch == 0,
            "Evaluate once per full pass; update budget must contain complete passes")
    device = torch.device("cpu")
    torch.set_num_threads(t["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    seed_all(t["seed"], device)
    generator = torch.Generator(device="cpu").manual_seed(t["seed"])
    dataset = TensorDataset(torch.from_numpy(bundle.X[train_indices]), torch.from_numpy(bundle.Y[train_indices]),
                            torch.from_numpy(train_indices.copy()))
    loader = DataLoader(dataset, batch_size=t["batch_size"], shuffle=True, generator=generator,
                        num_workers=0, drop_last=False, pin_memory=False)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # Weight-only initialization: neither old optimizer nor scheduler state is loaded.
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"],
                                  betas=tuple(t["betas"]), eps=t["eps"], foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda j:
        lr_factor(j, t["max_updates"], t["min_learning_rate"] / t["learning_rate"]))
    require(len(optimizer.state) == 0 and scheduler.last_epoch == 0, "Fresh optimizer/scheduler initialization failed")
    weights = torch.tensor(bundle.weights, dtype=torch.float32, device=device)
    output_scale = np.asarray(bundle.normalizer["output_scale"], dtype=np.float64)

    def evaluate_validation():
        model.eval()
        parts = []
        with torch.inference_mode():
            for start in range(0, len(val_indices), t["batch_size"]):
                ids = val_indices[start:start + t["batch_size"]]
                parts.append(model(torch.from_numpy(np.ascontiguousarray(bundle.X[ids]))).cpu().numpy())
        pred = decode_outputs(np.concatenate(parts), bundle.normalizer)
        metrics = summarize_validation(pred, bundle.Y_physical[val_indices], bundle.weights, output_scale,
                                       loss["lambda"], loss["delta"], cfg["comparison"]["relative_norm_threshold"])
        require(all(metrics["fields"][f]["relative_l2"]["undefined_count"] == 0 for f in FIELDS),
                "Undefined validation relative error; stop rather than changing comparison population")
        require(metrics["macro_mean_relative_l2"] is not None and math.isfinite(metrics["macro_mean_relative_l2"]), "Invalid selection score")
        return metrics, pred

    # Step zero is a real, user-run baseline evaluation, included in best selection.
    initial_metrics, initial_predictions = evaluate_validation()
    run_dir = new_run_dir(output_root, "finetune_" + cfg["variant"])
    environment = environment_info(device)
    environment.update(device="cpu", cpu_threads=torch.get_num_threads(), deterministic_algorithms=True)
    implementation = {"runner": digest(Path(__file__)), "objectives": digest(HERE / "paired_objectives.py")}
    implementation.update({name: digest(PROJECT / name) for name in cfg["expected_original_sources"]})
    experiment = dict(variant=cfg["variant"], objective=cfg["objective"],
        pair_signature=hashlib.sha256(canonical(shared_config(cfg)).encode()).hexdigest(),
        source_checkpoint_sha256=cfg["source_checkpoint_sha256"], source_epoch=cfg["source_epoch"],
        source_checkpoint=str(source), model_config=checkpoint["model_config"], data_fingerprints=bundle.fingerprints,
        normalizer_sha256=bundle.fingerprints["normalizer_sha256"], protocol=bundle.protocol,
        implementation_sha256=implementation, environment=environment, training=t, comparison=cfg["comparison"],
        loss=loss, reset_optimizer=True, reset_scheduler=True, resumable=False,
        source_config_sha256=digest(config_path), peer_config_sha256=digest(peer_path))
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "experiment.json", experiment)
    write_json(run_dir / "normalizer.json", bundle.normalizer)
    write_json(run_dir / "protocol.json", bundle.protocol)
    write_json(run_dir / "data_fingerprints.json", bundle.fingerprints)
    write_json(run_dir / "initial_validation.json", {"step": 0, "metrics": initial_metrics})
    base_payload = dict(format_version=1, operation="controlled_finetune", model_config=checkpoint["model_config"],
        normalizer=bundle.normalizer, protocol=bundle.protocol, data_fingerprints=bundle.fingerprints,
        training_config=copy.deepcopy(t), environment=environment, source_epoch=cfg["source_epoch"],
        source_checkpoint_sha256=cfg["source_checkpoint_sha256"], objective=cfg["objective"], loss_config=loss,
        pair_signature=experiment["pair_signature"], reset_optimizer=True, reset_scheduler=True, resumable=False,
        best_validation_metric=cfg["comparison"]["primary"], checkpoint_selection=cfg["comparison"])
    best_score = initial_metrics["macro_mean_relative_l2"]
    best_step, step = 0, 0
    order_hash, lr_hash = hashlib.sha256(), hashlib.sha256()

    def save_best(metrics, pred):
        save_checkpoint(run_dir / "best.pt", dict(base_payload, kind="best", model_state_dict=cpu_model_state(model),
            epoch=best_step // batches_per_epoch, fine_tune_step=best_step, best_epoch=best_step // batches_per_epoch,
            best_validation_loss=best_score, best_validation_score=best_score))
        write_json(run_dir / "best_validation.json", {"step": best_step, "metrics": metrics})
        np.savez_compressed(run_dir / "best_validation_predictions.npz", prediction_physical=pred,
            target_physical=bundle.Y_physical[val_indices], sample_id=bundle.sample_id[val_indices],
            material_id=bundle.material_id[val_indices], split=bundle.split[val_indices], x=bundle.x, weights=bundle.weights)

    def completion(status):
        return dict(status=status, completed_updates=step, planned_updates=t["max_updates"], best_step=best_step,
                    best_validation_score=best_score, best_validation_metric=cfg["comparison"]["primary"],
                    train_order_sha256=order_hash.hexdigest(), lr_sequence_sha256=lr_hash.hexdigest(),
                    test_evaluated=False, resumable=False, source_checkpoint_unchanged=digest(source) == cfg["source_checkpoint_sha256"])

    def log_row(metrics, epoch, train_mse=None, train_mixed=None, used_lr=None, seconds=0.):
        return dict(validation_record(step, metrics), epoch=epoch, learning_rate=used_lr,
            next_learning_rate=float(optimizer.param_groups[0]["lr"]), train_online_mse=train_mse,
            train_online_mixed_loss=train_mixed, best_step=best_step, best_validation_score=best_score,
            train_samples=0 if step == 0 else len(train_indices), validation_samples=len(val_indices), epoch_seconds=seconds)

    save_best(initial_metrics, initial_predictions)
    print(f"Pair {cfg['variant']}: CPU; {t['max_updates']} planned updates; output {run_dir}", flush=True)
    print("Weights from frozen best; optimizer/scheduler RESET; no resume and no test evaluation.", flush=True)
    first_row = log_row(initial_metrics, 0)
    try:
        with (run_dir / "history.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(first_row))
            writer.writeheader()
            writer.writerow(first_row)
            handle.flush()
            for epoch in range(1, t["max_updates"] // batches_per_epoch + 1):
                started = time.perf_counter()
                model.train()
                total_mse, total_mixed, sample_count = 0., 0., 0
                for x, y, indices in loader:
                    used_lr = float(optimizer.param_groups[0]["lr"])
                    expected_lr = t["learning_rate"] * lr_factor(step, t["max_updates"], t["min_learning_rate"] / t["learning_rate"])
                    require(math.isclose(used_lr, expected_lr, rel_tol=1e-13), "Learning-rate schedule drift")
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(x)
                    terms = loss_terms(prediction, y, weights, loss["lambda"], loss["delta"])
                    objective = terms[cfg["objective"]]
                    objective.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    require(bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients), "Nonfinite/missing gradients; no update performed")
                    optimizer.step()
                    step += 1
                    order_hash.update(canonical(bundle.sample_id[indices.numpy()].tolist()).encode() + b"\n")
                    lr_hash.update((used_lr.hex() + "\n").encode())
                    scheduler.step()
                    count = len(x)
                    total_mse += float(terms["mse"].detach()) * count
                    total_mixed += float(terms["mixed"].detach()) * count
                    sample_count += count
                require(sample_count == len(train_indices) and step == epoch * batches_per_epoch, "Incomplete epoch or incorrect update budget")
                metrics, pred = evaluate_validation()
                score = metrics["macro_mean_relative_l2"]
                if score < best_score:
                    best_score, best_step = score, step
                    save_best(metrics, pred)
                write_json(run_dir / "last_validation.json", {"step": step, "metrics": metrics})
                writer.writerow(log_row(metrics, epoch, total_mse / sample_count, total_mixed / sample_count,
                                        used_lr, time.perf_counter() - started))
                handle.flush()
                # Audit states only: this new mode intentionally offers no resume.
                save_checkpoint(run_dir / "last.pt", dict(base_payload, kind="finetune_last", model_state_dict=cpu_model_state(model),
                    epoch=epoch, fine_tune_step=step, best_epoch=best_step // batches_per_epoch, best_validation_loss=best_score,
                    best_validation_score=best_score, optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict()))
                print(f"{cfg['variant']} step {step}/{t['max_updates']}: validation mean-relative={score:.7g}, best={best_score:.7g}", flush=True)
        require(step == t["max_updates"], "Full paired budget was not completed")
        require(digest(source) == cfg["source_checkpoint_sha256"], "Frozen source checkpoint changed")
        write_json(run_dir / "completion.json", completion("completed"))
    except BaseException as exc:
        status = completion("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "completion.json", status)
        raise
    print(f"Completed paired arm {cfg['variant']}. No test evaluation. Best: {run_dir / 'best.pt'}", flush=True)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", required=True, help="config_A_mse.json or config_B_mixed.json, relative to this script")
    args = parser.parse_args(argv)
    run(args.config)


if __name__ == "__main__":
    main()
