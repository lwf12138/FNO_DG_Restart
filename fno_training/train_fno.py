"""User-run training only. No test evaluation and no execution on import."""
import argparse
import copy
import csv
import math
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .data import load_prepared
    from .metrics import weighted_mse
    from .model import build_model
    from .utils import (add_config_arguments, load_config, resolve_path, select_device, seed_all,
                        environment_info, new_run_dir, write_json, sha256_file, cpu_model_state,
                        save_checkpoint, load_checkpoint, capture_rng, restore_rng)
else:
    from data import load_prepared
    from metrics import weighted_mse
    from model import build_model
    from utils import (add_config_arguments, load_config, resolve_path, select_device, seed_all,
                       environment_info, new_run_dir, write_json, sha256_file, cpu_model_state,
                       save_checkpoint, load_checkpoint, capture_rng, restore_rng)


def validate_training_config(cfg):
    t = cfg["training"]
    for key in ("batch_size", "epochs", "early_stopping_patience"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer.")
    for key in ("num_workers", "scheduler_patience", "seed"):
        if type(t[key]) is not int or t[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer.")
    for key in ("learning_rate", "weight_decay", "min_learning_rate", "early_stopping_min_delta"):
        if not isinstance(t[key], (int, float)) or not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"training.{key} must be finite and nonnegative.")
    if t["learning_rate"] <= 0 or t["min_learning_rate"] > t["learning_rate"]:
        raise ValueError("Learning rate must be positive and min_learning_rate cannot exceed it.")
    if not 0 < t["scheduler_factor"] < 1:
        raise ValueError("scheduler_factor must lie strictly between 0 and 1.")


def make_loader(bundle, split, settings, generator=None):
    indices = bundle.indices(split)
    dataset = TensorDataset(torch.from_numpy(bundle.X[indices]), torch.from_numpy(bundle.Y[indices]))
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=split == "train",
                      num_workers=settings["num_workers"], drop_last=False, pin_memory=False,
                      generator=generator if split == "train" else None)


def run_epoch(model, loader, weights, device, optimizer=None):
    """Batch-mean loss is weighted by the actual batch size when aggregating."""
    training = optimizer is not None
    model.train(training)
    total_loss, samples = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = weighted_mse(prediction, y, weights)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite loss; aborting instead of saving invalid metrics.")
            if training:
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                    raise FloatingPointError("Missing or nonfinite parameter gradients; no optimizer step taken.")
                optimizer.step()
            count = x.shape[0]
            total_loss += float(loss.detach().cpu()) * count
            samples += count
    if samples != len(loader.dataset) or samples == 0:
        raise RuntimeError("Epoch did not consume every sample exactly once.")
    return total_loss / samples


def train(cfg, resume_path=None):
    validate_training_config(cfg)
    bundle = load_prepared(cfg)  # Does not prepare data or fit a normalizer.
    settings = cfg["training"]
    device = select_device(settings["device"])
    seed_all(settings["seed"], device)
    generator = torch.Generator(device="cpu").manual_seed(settings["seed"])
    train_loader = make_loader(bundle, "train", settings, generator)
    validation_loader = make_loader(bundle, "validation", settings)
    # No test DataLoader is constructed here.
    model = build_model(cfg["model"]).to(device)
    weights = torch.tensor(bundle.weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"], foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=settings["scheduler_factor"],
        patience=settings["scheduler_patience"], min_lr=settings["min_learning_rate"],
        threshold=0.0, threshold_mode="abs")
    start_epoch, best_epoch = 1, 0
    best_value, early_reference = math.inf, math.inf
    bad_epochs, best_state = 0, None
    resume_info = None
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path)
        if checkpoint.get("kind") != "last" or "optimizer_state_dict" not in checkpoint:
            raise ValueError("Resume requires last.pt, including optimizer/scheduler/RNG state.")
        for key, value in (("model_config", cfg["model"]), ("normalizer", bundle.normalizer),
                           ("protocol", bundle.protocol), ("data_fingerprints", bundle.fingerprints)):
            if checkpoint[key] != value:
                raise ValueError(f"Resume {key} differs from current data/model protocol.")
        for key, value in checkpoint["training_config"].items():
            if key != "epochs" and settings[key] != value:
                raise ValueError(f"Resume must preserve training.{key}; only total epochs may change.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_value, best_epoch = checkpoint["best_validation_loss"], checkpoint["best_epoch"]
        best_state = checkpoint["best_model_state_dict"]
        bad_epochs = checkpoint["early_stopping"]["bad_epochs"]
        early_reference = checkpoint["early_stopping"]["reference"]
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch > settings["epochs"]:
            raise ValueError("Resume has no remaining epochs; increase training.epochs explicitly.")
        if bad_epochs >= settings["early_stopping_patience"]:
            raise ValueError("This run already reached early stopping; resume will not bypass that decision.")
        restore_rng(checkpoint["rng_state"], generator, device)
        resume_info = {"checkpoint": str(resume_path), "sha256": sha256_file(resume_path),
                       "next_epoch": start_epoch}

    run_dir = new_run_dir(cfg["paths"]["results_dir"], "train")
    environment = environment_info(device)
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "normalizer.json", bundle.normalizer)
    write_json(run_dir / "data_fingerprints.json", bundle.fingerprints)
    write_json(run_dir / "protocol.json", bundle.protocol)
    write_json(run_dir / "environment.json", environment)
    if resume_info is not None:
        write_json(run_dir / "resume.json", resume_info)
    base_payload = dict(format_version=1, model_config=copy.deepcopy(cfg["model"]),
                        normalizer=bundle.normalizer, protocol=bundle.protocol,
                        data_fingerprints=bundle.fingerprints,
                        training_config=copy.deepcopy(settings), environment=environment)

    def best_payload():
        return dict(base_payload, kind="best", model_state_dict=best_state,
                    epoch=best_epoch, best_epoch=best_epoch, best_validation_loss=best_value)

    if best_state is not None:
        save_checkpoint(run_dir / "best.pt", best_payload())
    columns = ["epoch", "train_loss", "validation_loss", "learning_rate", "next_learning_rate",
               "best_validation_loss", "best_epoch", "bad_epochs", "train_samples", "validation_samples",
               "epoch_seconds"]
    print(f"Selected device: {device}; output: {run_dir}", flush=True)
    print("Training uses train/validation only. Test evaluation must be started separately.", flush=True)
    stop_reason = "maximum_epochs"
    with (run_dir / "history.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for epoch in range(start_epoch, settings["epochs"] + 1):
            started = time.perf_counter()
            used_lr = float(optimizer.param_groups[0]["lr"])
            training_loss = run_epoch(model, train_loader, weights, device, optimizer)
            validation_loss = run_epoch(model, validation_loader, weights, device)
            # best always tracks the strict minimum; min_delta affects early stopping only.
            if validation_loss < best_value:
                best_value, best_epoch = validation_loss, epoch
                best_state = cpu_model_state(model)
                save_checkpoint(run_dir / "best.pt", best_payload())
            if validation_loss < early_reference - settings["early_stopping_min_delta"]:
                early_reference, bad_epochs = validation_loss, 0
            else:
                bad_epochs += 1
            scheduler.step(validation_loss)
            record = dict(epoch=epoch, train_loss=training_loss, validation_loss=validation_loss,
                          learning_rate=used_lr, next_learning_rate=float(optimizer.param_groups[0]["lr"]),
                          best_validation_loss=best_value, best_epoch=best_epoch, bad_epochs=bad_epochs,
                          train_samples=len(train_loader.dataset), validation_samples=len(validation_loader.dataset),
                          epoch_seconds=time.perf_counter() - started)
            writer.writerow(record)
            handle.flush()
            payload = dict(base_payload, kind="last", model_state_dict=cpu_model_state(model),
                           epoch=epoch, best_epoch=best_epoch, best_validation_loss=best_value,
                           best_model_state_dict=best_state, optimizer_state_dict=optimizer.state_dict(),
                           scheduler_state_dict=scheduler.state_dict(),
                           early_stopping={"reference": early_reference, "bad_epochs": bad_epochs},
                           rng_state=capture_rng(generator, device))
            save_checkpoint(run_dir / "last.pt", payload)
            print(f"epoch {epoch}: train={training_loss:.7g}, validation={validation_loss:.7g}, "
                  f"best={best_value:.7g}, lr={used_lr:.4g}", flush=True)
            if bad_epochs >= settings["early_stopping_patience"]:
                stop_reason = "early_stopping"
                break
    write_json(run_dir / "completion.json", dict(stop_reason=stop_reason, last_epoch=epoch,
               best_epoch=best_epoch, best_validation_loss=best_value, test_evaluated=False))
    print(f"Finished. Best checkpoint: {run_dir / 'best.pt'}; test was not evaluated.")
    return run_dir


def latest_checkpoint(results_dir):
    """Only complete, atomically saved last.pt files can be resumed."""
    candidates = list(Path(results_dir).glob("train_*/last.pt"))
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path))) if candidates else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    resume_options = parser.add_mutually_exclusive_group()
    resume_options.add_argument("--resume", help="Own trusted last.pt; relative to the configuration directory.")
    resume_options.add_argument("--resume-latest", action="store_true",
                                help="Start if no saved run exists; otherwise continue the latest complete last.pt.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    for key in ("device", "epochs", "batch_size", "learning_rate", "seed"):
        value = getattr(args, key)
        if value is not None:
            cfg["training"][key] = value
    resume_path = resolve_path(args.resume, cfg)
    if args.resume_latest:
        resume_path = latest_checkpoint(cfg["paths"]["results_dir"])
        if resume_path is None:
            print("未找到已保存的训练进度，将开始新训练。", flush=True)
        else:
            checkpoint = load_checkpoint(resume_path)
            if checkpoint["early_stopping"]["bad_epochs"] >= checkpoint["training_config"]["early_stopping_patience"]:
                print(f"最近的训练已早停：{resume_path}\n如需独立新实验，请不带 --resume-latest 启动。", flush=True)
                return
            if checkpoint["epoch"] >= cfg["training"]["epochs"]:
                print(f"已完成第 {checkpoint['epoch']} 轮：{resume_path}\n"
                      "如需继续，请增大 config.json 的 training.epochs；本次没有启动训练。", flush=True)
                return
            print(f"继续已保存的训练：{resume_path}\n从第 {checkpoint['epoch'] + 1} 轮开始。", flush=True)
    try:
        train(cfg, resume_path)
    except KeyboardInterrupt:
        checkpoint_path = latest_checkpoint(cfg["paths"]["results_dir"])
        print("\n训练已停止。" + (f"已保存的进度：{checkpoint_path}" if checkpoint_path else
              "尚未完成第一轮，下次从第一轮开始。"), flush=True)
        print("再次运行同一个 --resume-latest 配置即可继续；未保存的当前轮会重跑。", flush=True)
    except RuntimeError as exc:
        raise RuntimeError(f"FNO training failed on explicitly requested {cfg['training']['device']!r}. "
                           "No device fallback was performed. Check the original error; if the backend "
                           "does not support complex FFT/gradients, choose cpu and run check_setup.py manually. "
                           f"Original error: {exc}") from exc


if __name__ == "__main__":
    main()
