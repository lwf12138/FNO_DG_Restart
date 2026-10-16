"""Manual data/model/device checks. Never creates an optimizer or trains."""
import argparse
import io
import json
import numpy as np

if __package__:
    from .data import load_prepared, decode_outputs
    from .utils import (add_config_arguments, load_config, select_device, environment_info,
                        cpu_model_state, resolve_path, write_json)
else:
    from data import load_prepared, decode_outputs
    from utils import (add_config_arguments, load_config, select_device, environment_info,
                       cpu_model_state, resolve_path, write_json)


def check(cfg, data_only=False, backward=False):
    if data_only and backward:
        raise ValueError("--data-only and --backward cannot be used together.")
    bundle = load_prepared(cfg)
    restored = decode_outputs(bundle.Y, bundle.normalizer)
    roundtrip_error = float(np.max(np.abs(restored - bundle.Y_physical)))
    if not np.allclose(restored, bundle.Y_physical, rtol=2e-6, atol=2e-7):
        raise ValueError("Output normalization round trip exceeds float32 storage tolerance.")
    report = dict(purpose="manual setup check; no optimizer, no training, no accuracy claim",
                  data_shapes={"X": list(bundle.X.shape), "Y": list(bundle.Y.shape)},
                  split_counts={s: len(bundle.indices(s)) for s in ("train", "validation", "test")},
                  normalization_max_absolute_roundtrip_error=roundtrip_error,
                  fingerprints=bundle.fingerprints, data_checks="passed",
                  forward="not executed", odd_length="not executed", gradients="not executed",
                  device="not executed", checkpoint="not executed", optimizer_step="never executed")
    if data_only:
        report["environment"] = environment_info()
        return report

    import torch
    if __package__:
        from .model import build_model
        from .metrics import weighted_mse
    else:
        from model import build_model
        from metrics import weighted_mse
    device = select_device(cfg["training"]["device"])
    report["environment"] = environment_info(device)
    report["availability"] = {"cuda": bool(torch.cuda.is_available()),
                              "mps_built": bool(torch.backends.mps.is_built()),
                              "mps_available": bool(torch.backends.mps.is_available())}
    model = build_model(cfg["model"]).to(device)
    model.eval()
    parameters = list(model.parameters())
    if any(p.device.type != device.type or (device.index is not None and p.device.index != device.index)
           for p in parameters):
        raise RuntimeError("Model parameters are not all on the selected device.")
    if any(p.dtype != (torch.complex64 if p.is_complex() else torch.float32) for p in parameters):
        raise RuntimeError("Expected float32 real weights and complex64 spectral weights.")
    batch = min(2, len(bundle.indices("train")))
    indices = bundle.indices("train")[:batch]
    x = torch.from_numpy(bundle.X[indices]).to(device)
    weights = torch.tensor(bundle.weights, dtype=torch.float32, device=device)
    observed_shapes = []

    def observe(module, args, output):
        observed_shapes.append((tuple(args[0].shape), tuple(output.shape)))

    handles = [conv.register_forward_hook(observe) for conv in model.operator.fno_blocks.convs]
    try:
        with torch.no_grad():
            predicted = model(x)
    finally:
        for handle in handles:
            handle.remove()
    if not torch.isfinite(predicted).all().item():
        raise FloatingPointError("Nonfinite model output.")
    nx = x.shape[-1]
    padded = nx + 2 * round(nx * cfg["model"]["domain_padding"])
    if len(observed_shapes) != cfg["model"]["n_layers"] or any(
            before[-1] != padded or after[-1] != padded for before, after in observed_shapes):
        raise RuntimeError("Unexpected spectral-layer padding/FFT lengths.")
    if model.operator.positional_embedding is not None:
        raise RuntimeError("A second coordinate embedding must not be appended.")
    modes = cfg["model"]["n_modes"][0] // 2 + 1
    if any(list(conv.n_modes) != [modes] for conv in model.operator.fno_blocks.convs):
        raise RuntimeError("Installed official spectral-mode convention differs from the pinned source.")
    report.update(forward="passed", output_shape=list(predicted.shape), actual_selected_device=str(parameters[0].device),
                  device="forward and parameter placement passed; optimizer compatibility not tested",
                  padded_nx=padded, retained_rfft_coefficients=modes,
                  parameter_count=sum(p.numel() for p in parameters))

    # This is a synthetic odd-length shape check only, not a resolution-accuracy experiment.
    odd_nx = nx if nx % 2 else nx + 1
    if odd_nx == nx:
        odd_prediction = predicted
    else:
        with torch.no_grad():
            odd_prediction = model(torch.zeros(batch, 5, odd_nx, dtype=torch.float32, device=device))
    if odd_prediction.shape != (batch, 3, odd_nx) or not torch.isfinite(odd_prediction).all().item():
        raise RuntimeError("Odd-length FNO input/output shape check failed.")
    report["odd_length"] = {"status": "passed", "input_nx": odd_nx, "output_nx": odd_prediction.shape[-1]}

    # Memory-only checkpoint probe: no best.pt, last.pt, fake metric or trained weight is created.
    probe = dict(kind="untrained_serialization_probe", model_state_dict=cpu_model_state(model),
                 model_config=cfg["model"], normalizer=bundle.normalizer, protocol=bundle.protocol)
    buffer = io.BytesIO()
    torch.save(probe, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)
    replica = build_model(loaded["model_config"]).to(device)
    replica.load_state_dict(loaded["model_state_dict"], strict=True)
    replica.eval()
    with torch.no_grad():
        reconstructed = replica(x)
    if not torch.allclose(predicted, reconstructed, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Memory-only checkpoint model round trip failed.")
    if loaded["normalizer"] != bundle.normalizer or loaded["protocol"] != bundle.protocol:
        raise RuntimeError("Checkpoint metadata round trip failed.")
    report["checkpoint"] = "memory-only state_dict/normalizer/protocol round trip passed; untrained"

    # Unit residual should have weighted normalized MSE=1 for any valid quadrature.
    unit_loss = weighted_mse(torch.ones_like(predicted), torch.zeros_like(predicted), weights)
    if not torch.allclose(unit_loss, torch.ones_like(unit_loss), rtol=1e-6, atol=1e-7):
        raise RuntimeError("Spatial/field MSE normalization check failed.")
    report["weighted_loss_normalization"] = "passed"
    if backward:
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(x)
        loss = weighted_mse(output, torch.zeros_like(output), weights)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None for g in gradients) or not all(torch.isfinite(g).all().item() for g in gradients):
            raise FloatingPointError("Missing or nonfinite real/complex parameter gradients.")
        report["gradients"] = "one explicitly requested backward passed; no optimizer exists and no update taken"
        model.zero_grad(set_to_none=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--device")
    parser.add_argument("--data-only", action="store_true", help="No model construction or device operations.")
    parser.add_argument("--backward", action="store_true", help="Explicitly request one gradient check; never updates weights.")
    parser.add_argument("--report", help="Optional new JSON report path, relative to the config directory.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.device is not None:
        cfg["training"]["device"] = args.device
    try:
        report = check(cfg, args.data_only, args.backward)
    except RuntimeError as exc:
        raise RuntimeError(f"Manual setup check failed on {cfg['training']['device']!r}. No fallback was used. "
                           f"For unsupported FFT/complex/device operations choose cpu explicitly. Original: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if args.report:
        path = resolve_path(args.report, cfg)
        if path.exists():
            raise FileExistsError(f"Will not overwrite an earlier check report: {path}")
        write_json(path, report)


if __name__ == "__main__":
    main()
