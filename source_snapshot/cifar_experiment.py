"""End-to-end CIFAR-10 Early-Bird detection and t_s sweep runner."""
import csv
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MPL_CONFIG = ROOT / ".cache" / "matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import torch
import yaml

from analysis.plot_mask_distance import plot_mask_distance
from analysis.plot_ts_comparison import plot_ts_comparison
from data.cifar10 import create_loaders
from earlybird.stability_detector import StabilityDetector
from evaluation.classification import evaluate_classifier
from evaluation.flops import count_dense_flops
from evaluation.inference_time import measure_inference
from evaluation.model_size import model_size_bytes
from evaluation.parameter_count import parameter_count
from models.cifar_vgg import create_vgg16_bn, structurally_prune_vgg16
from pruning.channel_manager import BNChannelMaskManager
from pruning.mask_generator import bn_channel_masks
from pruning.mask_utils import load_masks, save_masks
from training.builders import build_optimizer, build_scheduler
from training.trainer import train_epoch
from utils.seed import set_seed


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_output(exp_id, config):
    root = ROOT / config["experiment"].get("output_root", "outputs") / exp_id
    root.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "masks", "plots"):
        (root / name).mkdir()
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
           "device": "cuda:0" if torch.cuda.is_available() else "cpu",
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (root / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return root


def _append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _save_summary(out, summary):
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def _write_final_metrics_to_epoch_csv(metrics_path, summary):
    """Add run-level test and compact-model metrics to every epoch row."""
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return
    with metrics_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    # These were per-epoch placeholders and were always empty: FLOPs, size and
    # inference latency are measured once on the final compact model instead.
    fieldnames = [name for name in fieldnames if name not in {"flops", "inference_time", "model_size"}]
    final_values = {
        "run_status": summary.get("status"),
        "final_test_loss": summary.get("loss"),
        "final_test_accuracy": summary.get("accuracy"),
        "final_test_f1_macro": summary.get("f1_macro"),
        "final_parameter_count": summary.get("parameter_count"),
        "final_zero_parameter_count": summary.get("zero_parameter_count"),
        "final_parameter_sparsity": summary.get(
            "parameter_sparsity", summary.get("sparsity")),
        "final_channel_sparsity": summary.get("channel_sparsity"),
        "final_flops_per_sample": summary.get("flops_per_sample"),
        "final_model_size_bytes": summary.get("model_size_bytes"),
        "final_inference_seconds_per_sample": summary.get("inference_seconds_per_sample"),
        "run_training_seconds": summary.get("training_seconds"),
        "run_wall_seconds": summary.get("wall_seconds"),
        "mask_detection_seconds": summary.get("mask_detection_seconds"),
        "mask_generation_seconds": summary.get("mask_generation_seconds"),
        "compaction_seconds": summary.get("compaction_seconds"),
        "final_compaction_seconds": summary.get("final_compaction_seconds"),
        "validation_seconds": summary.get("validation_seconds"),
        "checkpoint_seconds": summary.get("checkpoint_seconds"),
        "evaluation_seconds": summary.get("evaluation_seconds"),
        "end_to_end_seconds": summary.get("end_to_end_seconds"),
        "paired_mask_match": summary.get("paired_mask_match"),
    }
    reduction_fields = {
        "final_parameter_count_reduction_percent": "parameter_count_reduction_percent",
        "final_flops_reduction_percent": "flops_per_sample_reduction_percent",
        "final_model_size_reduction_percent": "model_size_bytes_reduction_percent",
    }
    for output_name, summary_name in reduction_fields.items():
        if summary_name in summary:
            final_values[output_name] = summary[summary_name]
    for name in final_values:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        for name in ("flops", "inference_time", "model_size"):
            row.pop(name, None)
        row.update({name: "" if value is None else value for name, value in final_values.items()})
    tmp_path = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(metrics_path)


def _make_loaders(config):
    data_root = Path(config["data"].get("root", "data/cifar10"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    return create_loaders(root=str(data_root),
        batch_size=int(config["data"]["kwargs"]["batch_size"]),
        seed=int(config["project"]["seed"]))


def run_one(config, exp_id, stage, ts=None, information_mode=None, stable_masks=None,
            detect=False, reference_tm=None, device=None):
    set_seed(int(config["project"]["seed"]), deterministic=False)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = _prepare_output(exp_id, config)
    (out / "run.log").write_text(f"stage={stage} ts={ts} mode={information_mode}\n", encoding="utf-8")
    loaders = _make_loaders(config)
    model = create_vgg16_bn(num_classes=10).to(device)
    torch.save(model.state_dict(), out / "checkpoints" / "initial.pt")
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    criterion = torch.nn.CrossEntropyLoss()
    detector = StabilityDetector(float(config["earlybird"]["epsilon"]),
                                 int(config["earlybird"]["mask_window"])) if detect else None
    mask_manager = None
    chosen_masks = None
    transition_done = ts is None
    start = time.perf_counter()
    train_seconds_total = 0.0
    detection_seconds_total = 0.0
    if device.type == "cuda":
        _sync(device)
    for epoch in range(1, int(config["experiment"]["epochs"]) + 1):
        # Match the reference loop: inspect the current weights before this epoch's updates.
        if detector:
            mask_start = time.perf_counter()
            candidate = bn_channel_masks(model, float(config["pruning"]["selected_ratio"]))
            became_stable = detector.update(epoch - 1, candidate)
            detection_seconds_total += time.perf_counter() - mask_start
            if detector.last_distances:
                for compared_index, distance in enumerate(detector.last_distances):
                    _append_csv(out / "mask_distances.csv", {
                        "epoch": epoch - 1, "compared_mask_offset": compared_index + 1,
                        "mask_distance": distance})
            if became_stable:
                stable_masks = candidate
                save_masks(stable_masks, out / "masks" / "stable_mask.pt")
                torch.save({"epoch": epoch - 1, "model": model.state_dict()},
                           out / "checkpoints" / "tm.pt")
                with (out / "run.log").open("a", encoding="utf-8") as log:
                    log.write(f"stage={stage} detected_tm={epoch - 1} epsilon={detector.threshold} "
                              f"mask_window={detector.mask_window}\n")
        if not transition_done and epoch - 1 == ts:
            if information_mode == "online":
                mask_start = time.perf_counter()
                chosen_masks = bn_channel_masks(model, float(config["pruning"]["selected_ratio"]))
                detection_seconds_total += time.perf_counter() - mask_start
            else:
                chosen_masks = stable_masks
            if chosen_masks is None:
                raise RuntimeError("Posthoc run has no detected stable mask")
            if any(int(mask.sum()) == 0 for mask in chosen_masks.values()):
                raise RuntimeError("Selected channel mask removes every channel in a BN layer")
            mask_manager = BNChannelMaskManager(model, chosen_masks)
            save_masks(chosen_masks, out / "masks" / "transition_mask.pt")
            with (out / "run.log").open("a", encoding="utf-8") as log:
                log.write(f"transition_applied after_epoch={ts} mode={information_mode} "
                          f"channel_sparsity={mask_manager.sparsity():.6f}\n")
            transition_done = True

        _sync(device)
        epoch_start = time.perf_counter()
        train_result = train_epoch(model, loaders["train"], optimizer, criterion, device)
        _sync(device)
        train_seconds = time.perf_counter() - epoch_start
        train_seconds_total += train_seconds
        validation = evaluate_classifier(model, loaders["validation"], device, labels=list(range(10)))
        row = {"experiment_id": exp_id, "seed": config["project"]["seed"], "epoch": epoch,
            "ts": ts, "tm": reference_tm, "information_mode": information_mode,
            "mask_distance": None, "train_loss": train_result["train_loss"],
            "validation_loss": validation["loss"], "accuracy": validation["accuracy"],
            "f1_macro": validation["f1_macro"], "training_seconds": train_seconds,
            "sparsity": mask_manager.sparsity() if mask_manager else 0.0,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "flops": None, "inference_time": None, "model_size": None}
        _append_csv(out / "metrics.csv", row)
        progress = (f"{exp_id} epoch {epoch}/{config['experiment']['epochs']} "
                    f"train_loss={train_result['train_loss']:.4f} "
                    f"val_acc={validation['accuracy']:.4f} train_s={train_seconds:.1f}")
        print(progress, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as log:
            log.write(progress + "\n")

        scheduler.step()

    _sync(device)
    training_seconds = time.perf_counter() - start
    if ts is not None and not transition_done:
        raise RuntimeError(f"t_s={ts} transition was not reached")
    if mask_manager:
        compact = structurally_prune_vgg16(model, chosen_masks).to(device).eval()
        mask_manager.close()
    else:
        compact = model.eval()
    test_metrics = evaluate_classifier(compact, loaders["test"], device, labels=list(range(10)))
    sample = next(iter(loaders["test"]))[0][:1].to(device)
    inference = measure_inference(compact, sample, device, warmup=10, repeats=100)
    flops = count_dense_flops(compact, tuple(sample.shape), device)
    counts = parameter_count(compact)
    checkpoint_path = out / "checkpoints" / "final.pt"
    torch.save(compact.state_dict(), checkpoint_path)
    summary = {"experiment_id": exp_id, "status": "completed", "stage": stage,
        "dataset": "CIFAR-10", "model": "CIFAR VGG16-BN", "seed": config["project"]["seed"],
        "device": str(device), "tm": detector.tm if detector else reference_tm,
        "tm_detected": detector.detected if detector else None, "ts": ts,
        "information_mode": information_mode, **test_metrics,
        "training_seconds": train_seconds_total, "wall_seconds": training_seconds,
        "mask_detection_seconds": detection_seconds_total,
        "inference_seconds_per_sample": inference["mean_seconds"],
        "inference_repeats": inference["repeats"], "parameter_count": counts["parameter_count"],
        "zero_parameter_count": counts["zero_parameter_count"], "sparsity": counts["sparsity"],
        "flops_per_sample": flops, "model_size_bytes": model_size_bytes(checkpoint_path),
        "checkpoint": str(checkpoint_path)}
    _save_summary(out, summary)
    if detector and (out / "mask_distances.csv").exists():
        plot_mask_distance(out / "mask_distances.csv", out / "plots" / "mask_distance.png", detector.tm)
    if detector and detector.tm is not None:
        with (out / "metrics.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for metric_row in rows:
            metric_row["tm"] = detector.tm
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    _write_final_metrics_to_epoch_csv(out / "metrics.csv", summary)
    return summary, stable_masks


def run_full(config, resume_baseline=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required to run this full 160-epoch experiment in a practical time")
    torch.backends.cudnn.benchmark = True
    if resume_baseline:
        baseline_dir = Path(resume_baseline)
        if not baseline_dir.is_absolute():
            baseline_dir = ROOT / baseline_dir
        baseline = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
        if baseline.get("status") != "completed" or not baseline.get("tm_detected"):
            raise ValueError("Resume baseline must be completed and contain a detected t_m")
        stable_masks = load_masks(baseline_dir / "masks" / "stable_mask.pt")
        marker = "_baseline_seed1"
        if not baseline["experiment_id"].endswith(marker):
            raise ValueError("Baseline experiment folder name does not contain the run marker")
        run_id = baseline["experiment_id"][:-len(marker)]
    else:
        run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
        prefix = run_id + "_"
        # A single dense run supplies both the baseline and the non-invasive t_m trace.
        baseline, stable_masks = run_one(config, prefix + "baseline_seed1", "dense_baseline_with_tm_detection",
            detect=True, device=torch.device("cuda:0"))
    prefix = run_id + "_"
    if stable_masks is None:
        raise RuntimeError("Early-Bird mask was not detected; t_s posthoc comparisons cannot proceed")
    detector_summary = baseline
    tm = detector_summary["tm"]
    offsets = config["ts"]["relative_to_tm"]
    times = {"C1_before": max(1, tm - int(offsets["before_offset"])),
             "C2_near": max(1, tm + int(offsets["near_offset"])),
             "C3_after": max(1, tm + int(offsets["after_offset"])),
             "C4_fixed": int(config["ts"]["fixed_epoch"])}
    summaries = []
    for mode in config["ts"]["information_modes"]:
        for label, ts_value in times.items():
            summary, _ = run_one(config, prefix + f"{label}_{mode}_seed1", "ts_comparison",
                ts=ts_value, information_mode=mode,
                stable_masks=stable_masks if mode == "posthoc" else None,
                reference_tm=tm,
                device=torch.device("cuda:0"))
            summary["condition"] = label
            summaries.append(summary)
            for metric, base_metric in (("parameter_count", "parameter_count"),
                                        ("flops_per_sample", "flops_per_sample"),
                                        ("model_size_bytes", "model_size_bytes")):
                base = baseline.get(base_metric)
                value = summary.get(metric)
                summary[metric + "_reduction_percent"] = (
                    (base - value) / base * 100 if base and value is not None else None)
            out = ROOT / config["experiment"].get("output_root", "outputs") / summary["experiment_id"]
            _write_final_metrics_to_epoch_csv(out / "metrics.csv", summary)
            _save_summary(out, summary)
    comparison_path = ROOT / config["experiment"].get("output_root", "outputs") / f"ts_comparison_{run_id}.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    try:
        plot_ts_comparison(comparison_path, comparison_path.with_suffix(".png"))
    except (ValueError, TypeError):
        pass
    run_summary = {"run_id": run_id, "baseline": baseline, "detection": detector_summary,
            "conditions": summaries, "comparison_csv": str(comparison_path)}
    (comparison_path.parent / f"run_summary_{run_id}.json").write_text(
        json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary
