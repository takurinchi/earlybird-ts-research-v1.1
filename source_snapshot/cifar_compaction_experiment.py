"""Matched Dense-control and train-time-compaction CIFAR-10 experiments."""
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch

from analysis.plot_compaction_comparison import plot_compaction_comparison
from cifar_experiment import (
    ROOT, _append_csv, _make_loaders, _prepare_output, _save_summary,
    _sync, _write_final_metrics_to_epoch_csv, run_one,
)
from evaluation.classification import evaluate_classifier
from evaluation.flops import count_dense_flops
from evaluation.inference_time import measure_inference
from evaluation.model_size import model_size_bytes
from evaluation.parameter_count import parameter_count
from evaluation.structured_metrics import (
    estimate_vgg16_compact_flops, estimate_vgg16_compact_parameters,
)
from models.cifar_vgg import structurally_prune_vgg16
from pruning.channel_manager import BNChannelMaskManager
from pruning.mask_generator import bn_channel_masks
from pruning.mask_utils import load_masks, save_masks
from training.builders import build_optimizer, build_scheduler
from training.optimizer_state import migrate_optimizer_state, migrate_scheduler_state
from training.trainer import train_epoch
from utils.seed import set_seed


def _mask_digest(masks):
    digest = hashlib.sha256()
    for name in sorted(masks):
        digest.update(name.encode("utf-8"))
        digest.update(masks[name].detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _same_masks(left, right):
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[name].detach().cpu().bool(), right[name].detach().cpu().bool())
               for name in left)


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _checkpoint(model, path, device):
    _sync(device)
    started = time.perf_counter()
    torch.save(model.state_dict(), path)
    _sync(device)
    return time.perf_counter() - started


def _load_baseline(config, resume_baseline, run_id, device):
    seed = int(config["project"]["seed"])
    if resume_baseline:
        baseline_dir = Path(resume_baseline)
        if not baseline_dir.is_absolute():
            baseline_dir = ROOT / baseline_dir
        summary_path = baseline_dir / "summary.json"
        mask_path = baseline_dir / "masks" / "stable_mask.pt"
        if not mask_path.is_file():
            mask_path = baseline_dir / "stable_mask.pt"
        if not summary_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                "Resume baseline needs summary.json and stable_mask.pt (in masks/ or at the run root)")
        baseline = json.loads(summary_path.read_text(encoding="utf-8"))
        if baseline.get("status") != "completed" or not baseline.get("tm_detected"):
            raise ValueError("Resume baseline must be completed and have a detected t_m")
        if int(baseline.get("seed", -1)) != seed:
            raise ValueError("Resume baseline seed does not match the Ver.1.1 seed")
        source_config_path = baseline_dir / "config.yaml"
        if source_config_path.exists():
            import yaml
            source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8")) or {}
            source_ratio = source_config.get("pruning", {}).get("selected_ratio")
            current_ratio = config.get("pruning", {}).get("selected_ratio")
            if source_ratio is not None and float(source_ratio) != float(current_ratio):
                raise ValueError("Resume baseline pruning ratio does not match the current config")
        masks = load_masks(mask_path)
        return baseline, masks, baseline.get("experiment_id"), run_id, baseline_dir

    baseline_id = f"{run_id}_baseline_seed{seed}"
    baseline, masks = run_one(
        config, baseline_id, "dense_baseline_with_tm_detection",
        detect=True, device=device)
    if not baseline.get("tm_detected") or masks is None:
        raise RuntimeError("Early-Bird mask was not detected; Ver.1.1 comparisons cannot proceed")
    return baseline, masks, baseline_id, run_id, ROOT / config["experiment"].get("output_root", "outputs") / baseline_id


def _run_condition_impl(config, experiment_id, strategy, information_mode, ts, tm,
                        baseline_masks, mask_source_id, transition_checkpoint,
                        dense_baseline_metrics):
    """Run one comparison arm, compacting at t_s only for the treatment arm."""
    seed = int(config["project"]["seed"])
    deterministic = bool(config.get("reproducibility", {}).get("deterministic", True))
    set_seed(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and config.get("runtime", {}).get("require_cuda", True):
        raise RuntimeError("CUDA-enabled PyTorch is required for the configured full experiment")
    torch.backends.cudnn.benchmark = not deterministic

    run_started = time.perf_counter()
    out = _prepare_output(experiment_id, config)
    environment_path = out / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment.update({"deterministic_requested": deterministic,
                        "training_strategy": strategy,
                        "information_mode": information_mode})
    _json_write(environment_path, environment)
    (out / "run.log").write_text(
        f"stage=ts_compaction strategy={strategy} ts={ts} mode={information_mode}\n",
        encoding="utf-8")

    loaders = _make_loaders(config)
    # Keep the model choice explicit and fail clearly for unsupported architectures.
    from models.cifar_vgg import create_vgg16_bn
    if config["experiment"].get("model", {}).get("factory", "models.cifar_vgg:create_vgg16_bn") != \
            "models.cifar_vgg:create_vgg16_bn":
        raise ValueError("Ver.1.1 currently supports only models.cifar_vgg:create_vgg16_bn")
    model = create_vgg16_bn(**config["experiment"].get("model", {}).get("kwargs", {"num_classes": 10})).to(device)

    checkpoint_policy = config["measurement"]["checkpoint_policy"]
    if not checkpoint_policy.get("initial", False) or not checkpoint_policy.get("final", False):
        raise ValueError("Ver.1.1 requires one initial and one final checkpoint for every run")
    checkpoint_seconds = _checkpoint(model, out / "checkpoints" / "initial.pt", device)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])
    criterion = torch.nn.CrossEntropyLoss()
    chosen_masks = None
    mask_manager = None
    transition_done = False
    params_before_transition = None
    params_after_transition = None
    params_actual_after_transition = None
    flops_before_transition = None
    flops_after_transition = None
    structural_metric_seconds = 0.0
    transition_channel_sparsity = None
    mask_generation_seconds = 0.0
    compaction_seconds = 0.0
    dense_train_seconds = 0.0
    post_ts_train_seconds = 0.0
    training_seconds = 0.0
    validation_seconds = 0.0

    epochs = int(config["experiment"]["epochs"])
    if not 1 <= int(ts) < epochs:
        raise ValueError(f"t_s must be between 1 and epochs-1; got t_s={ts}, epochs={epochs}")
    transition_checkpoint_path = out / "checkpoints" / "transition_before.pt"
    loop_started = time.perf_counter()
    if device.type == "cuda":
        _sync(device)

    for epoch in range(1, epochs + 1):
        if not transition_done and epoch - 1 == int(ts):
            if transition_checkpoint:
                checkpoint_seconds += _checkpoint(model, transition_checkpoint_path, device)
            params_before_transition = sum(parameter.numel() for parameter in model.parameters())

            if information_mode == "online":
                if device.type == "cuda":
                    _sync(device)
                mask_started = time.perf_counter()
                chosen_masks = bn_channel_masks(
                    model, float(config["pruning"]["selected_ratio"]))
                if device.type == "cuda":
                    _sync(device)
                mask_generation_seconds += time.perf_counter() - mask_started
                source_run = experiment_id
            elif information_mode == "posthoc":
                chosen_masks = {name: mask.detach().clone() for name, mask in baseline_masks.items()}
                source_run = mask_source_id
            else:
                raise ValueError(f"Unsupported information_mode: {information_mode}")

            if any(int(mask.sum()) == 0 for mask in chosen_masks.values()):
                raise ValueError("Selected channel mask removes every channel in a BN layer")
            save_masks(chosen_masks, out / "masks" / "transition_mask.pt")
            mask_hash = _mask_digest(chosen_masks)
            transition_channel_sparsity = 1.0 - (
                sum(int(mask.sum()) for mask in chosen_masks.values()) /
                sum(mask.numel() for mask in chosen_masks.values()))

            if device.type == "cuda":
                _sync(device)
            structural_metrics_started = time.perf_counter()
            configured_shape = tuple(config["measurement"]["evaluation"]["input_shape"])
            flops_before_transition = count_dense_flops(
                model, input_shape=configured_shape, device=device) / configured_shape[0]
            flops_after_transition = estimate_vgg16_compact_flops(
                model, chosen_masks, input_shape=configured_shape) / configured_shape[0]
            params_after_transition = estimate_vgg16_compact_parameters(model, chosen_masks)
            if device.type == "cuda":
                _sync(device)
            structural_metric_seconds += time.perf_counter() - structural_metrics_started

            if strategy == "dense_late_compaction":
                mask_manager = BNChannelMaskManager(model, chosen_masks)
            elif strategy == "train_time_compaction":
                if device.type == "cuda":
                    _sync(device)
                compact_started = time.perf_counter()
                # Module constructors initialize temporary tensors. Preserve RNG so
                # paired arms consume the same augmentation and sampler sequence.
                with torch.random.fork_rng(devices=[]):
                    compact, parameter_slices, channel_indices = structurally_prune_vgg16(
                        model, chosen_masks, return_parameter_slices=True)
                compact = compact.to(device)
                new_optimizer = build_optimizer(compact, config["training"])
                migrate_optimizer_state(
                    optimizer, new_optimizer, model, compact, parameter_slices)
                new_scheduler = build_scheduler(new_optimizer, config["training"])
                migrate_scheduler_state(scheduler, new_scheduler, optimizer, new_optimizer)
                if device.type == "cuda":
                    _sync(device)
                compaction_seconds += time.perf_counter() - compact_started
                model, optimizer, scheduler = compact, new_optimizer, new_scheduler
                params_actual_after_transition = sum(
                    parameter.numel() for parameter in model.parameters())
                if params_actual_after_transition != params_after_transition:
                    raise RuntimeError(
                        "Projected and actual compact parameter counts do not match")
                indices_to_save = {name: indices.tolist() for name, indices in channel_indices.items()}
            else:
                raise ValueError(f"Unsupported training_strategy: {strategy}")

            if strategy == "dense_late_compaction":
                params_actual_after_transition = params_before_transition
                indices_to_save = {
                    name: torch.where(mask.detach().cpu().bool())[0].tolist()
                    for name, mask in chosen_masks.items()}
                indices_to_save["classifier.input"] = indices_to_save[list(chosen_masks)[-1]]
            _json_write(out / "masks" / "channel_indices.json", indices_to_save)

            with (out / "run.log").open("a", encoding="utf-8") as log:
                log.write(
                    f"transition_after_epoch={ts} first_compact_epoch={ts + 1} "
                    f"strategy={strategy} channel_sparsity={transition_channel_sparsity:.8f} "
                    f"mask_source_run_id={source_run} mask_sha256={mask_hash} "
                    f"parameters_before={params_before_transition} "
                    f"projected_parameters_after={params_after_transition} "
                    f"actual_parameters_after={params_actual_after_transition} "
                    f"flops_before={flops_before_transition} "
                    f"projected_flops_after={flops_after_transition}\n")
            transition_done = True

        _sync(device)
        epoch_train_started = time.perf_counter()
        train_result = train_epoch(model, loaders["train"], optimizer, criterion, device)
        _sync(device)
        epoch_training_seconds = time.perf_counter() - epoch_train_started
        training_seconds += epoch_training_seconds
        if epoch <= int(ts):
            dense_train_seconds += epoch_training_seconds
        else:
            post_ts_train_seconds += epoch_training_seconds

        _sync(device)
        validation_started = time.perf_counter()
        validation_labels = [int(value) for value in config["measurement"]["evaluation"]["labels"]]
        validation = evaluate_classifier(
            model, loaders["validation"], device, labels=validation_labels)
        _sync(device)
        epoch_validation_seconds = time.perf_counter() - validation_started
        validation_seconds += epoch_validation_seconds

        row = {
            "experiment_id": experiment_id,
            "training_strategy": strategy,
            "information_mode": information_mode,
            "seed": seed,
            "epoch": epoch,
            "tm": tm,
            "ts": int(ts),
            "transition_after_epoch": int(ts),
            "first_compact_epoch": int(ts) + 1,
            "model_phase": "dense" if epoch <= int(ts) or strategy == "dense_late_compaction" else "compact",
            "train_loss": train_result["train_loss"],
            "train_accuracy": train_result["train_accuracy"],
            "validation_loss": validation["loss"],
            "accuracy": validation["accuracy"],
            "f1_macro": validation["f1_macro"],
            "training_seconds": epoch_training_seconds,
            "validation_seconds": epoch_validation_seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "channel_sparsity": transition_channel_sparsity if transition_done else 0.0,
            # Scanning every CUDA parameter tensor for exact zeros here would add
            # architecture-dependent overhead to the timing comparison.
            "parameter_sparsity": None,
            "transition_applied": int(transition_done and epoch == int(ts) + 1),
        }
        _append_csv(out / "metrics.csv", row)
        with (out / "run.log").open("a", encoding="utf-8") as log:
            log.write(
                f"epoch={epoch}/{epochs} phase={row['model_phase']} "
                f"train_loss={train_result['train_loss']:.6f} "
                f"val_accuracy={validation['accuracy']:.6f} "
                f"lr={row['learning_rate']:.8g} "
                f"train_s={epoch_training_seconds:.4f} "
                f"validation_s={epoch_validation_seconds:.4f} "
                f"parameters={row['parameter_count']}\n")
        print(
            f"{experiment_id} epoch {epoch}/{epochs} phase={row['model_phase']} "
            f"val_acc={validation['accuracy']:.4f} train_s={epoch_training_seconds:.2f}",
            flush=True)
        scheduler.step()

    _sync(device)
    wall_seconds = time.perf_counter() - loop_started
    if not transition_done:
        raise RuntimeError(f"t_s={ts} transition was not reached")

    final_compaction_seconds = 0.0
    if strategy == "dense_late_compaction":
        if device.type == "cuda":
            _sync(device)
        final_compact_started = time.perf_counter()
        with torch.random.fork_rng(devices=[]):
            final_model, _, _ = structurally_prune_vgg16(
                model, chosen_masks, return_parameter_slices=True)
        final_model = final_model.to(device).eval()
        if device.type == "cuda":
            _sync(device)
        final_compaction_seconds = time.perf_counter() - final_compact_started
        if mask_manager:
            mask_manager.close()
    else:
        final_model = model.eval()

    evaluation_started = time.perf_counter()
    eval_settings = config["measurement"]["evaluation"]
    labels = [int(value) for value in eval_settings["labels"]]
    test_metrics = evaluate_classifier(final_model, loaders["test"], device, labels=labels)
    sample_batch = next(iter(loaders["test"]))[0]
    inference_batch_size = int(eval_settings["inference_batch_size"])
    sample = sample_batch[:inference_batch_size].to(device)
    if list(sample.shape) != list(eval_settings["input_shape"]):
        raise ValueError(
            f"Configured inference input shape {eval_settings['input_shape']} "
            f"does not match dataset sample shape {list(sample.shape)}")
    inference = measure_inference(
        final_model, sample, device,
        warmup=int(eval_settings["warmup"]), repeats=int(eval_settings["repeats"]))
    flops = count_dense_flops(final_model, tuple(sample.shape), device) / inference_batch_size
    final_counts = parameter_count(final_model)
    if device.type == "cuda":
        _sync(device)
    evaluation_seconds = time.perf_counter() - evaluation_started

    final_checkpoint_seconds = _checkpoint(final_model, out / "checkpoints" / "final.pt", device)
    checkpoint_seconds += final_checkpoint_seconds
    final_size = model_size_bytes(out / "checkpoints" / "final.pt")
    baseline_values = dense_baseline_metrics
    summary = {
        "experiment_id": experiment_id,
        "status": "completed",
        "stage": "ts_compaction_comparison",
        "dataset": "CIFAR-10",
        "model": "CIFAR VGG16-BN",
        "seed": seed,
        "device": str(device),
        "tm": int(tm),
        "ts": int(ts),
        "transition_after_epoch": int(ts),
        "first_compact_epoch": int(ts) + 1,
        "training_strategy": strategy,
        "information_mode": information_mode,
        "mask_source_run_id": source_run,
        "mask_sha256": mask_hash,
        "channel_sparsity": transition_channel_sparsity,
        "parameter_count_before_compaction": params_before_transition,
        "parameter_count_after_compaction": params_after_transition,
        "parameter_count_actual_after_transition": params_actual_after_transition,
        "flops_before_compaction": flops_before_transition,
        "flops_after_compaction": flops_after_transition,
        "parameter_count": final_counts["parameter_count"],
        "zero_parameter_count": final_counts["zero_parameter_count"],
        "parameter_sparsity": final_counts["sparsity"],
        "flops_per_sample": flops,
        "model_size_bytes": final_size,
        **test_metrics,
        "dense_train_seconds": dense_train_seconds,
        "compact_train_seconds": post_ts_train_seconds,
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "mask_generation_seconds": mask_generation_seconds,
        "compaction_seconds": compaction_seconds,
        "structural_metric_seconds": structural_metric_seconds,
        "final_compaction_seconds": final_compaction_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": wall_seconds,
        "inference_seconds_per_sample": inference["mean_seconds"] / inference_batch_size,
        "inference_repeats": inference["repeats"],
        "inference_batch_size": inference_batch_size,
        "class_labels": labels,
        "checkpoint_format": eval_settings["checkpoint_format"],
        "checkpoint": "checkpoints/final.pt",
        "transition_checkpoint_saved": bool(transition_checkpoint),
    }
    for metric_name, baseline_name in (
        ("parameter_count_reduction_percent", "parameter_count"),
        ("flops_per_sample_reduction_percent", "flops_per_sample"),
        ("model_size_bytes_reduction_percent", "model_size_bytes"),
    ):
        baseline_value = baseline_values.get(baseline_name)
        compact_value = summary.get(baseline_name)
        summary[metric_name] = (
            (baseline_value - compact_value) / baseline_value * 100.0
            if baseline_value and compact_value is not None else None)

    summary["end_to_end_seconds"] = time.perf_counter() - run_started
    _save_summary(out, summary)
    _write_final_metrics_to_epoch_csv(out / "metrics.csv", summary)
    return summary, chosen_masks, out


def _run_condition(config, experiment_id, strategy, information_mode, ts, tm,
                   baseline_masks, mask_source_id, transition_checkpoint,
                   dense_baseline_metrics):
    try:
        return _run_condition_impl(
            config, experiment_id, strategy, information_mode, ts, tm,
            baseline_masks, mask_source_id, transition_checkpoint,
            dense_baseline_metrics)
    except Exception as exc:
        output_root = ROOT / config["experiment"].get("output_root", "outputs")
        out = output_root / experiment_id
        if not out.exists():
            out = _prepare_output(experiment_id, config)
        log_path = out / "run.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"status=failed error_type={type(exc).__name__} error={exc}\n")
        metrics_path = out / "metrics.csv"
        completed_epochs = 0
        if metrics_path.is_file() and metrics_path.stat().st_size:
            with metrics_path.open(newline="", encoding="utf-8") as f:
                completed_epochs = sum(1 for _ in csv.DictReader(f))
        failed = {
            "experiment_id": experiment_id,
            "status": "failed",
            "stage": "ts_compaction_comparison",
            "training_strategy": strategy,
            "information_mode": information_mode,
            "seed": int(config["project"]["seed"]),
            "tm": int(tm),
            "ts": int(ts),
            "mask_source_run_id": mask_source_id,
            "completed_epochs": completed_epochs,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "accuracy": None,
            "f1_macro": None,
            "parameter_count": None,
            "flops_per_sample": None,
            "model_size_bytes": None,
            "dense_train_seconds": None,
            "compact_train_seconds": None,
            "training_seconds": None,
            "validation_seconds": None,
            "compaction_seconds": None,
            "checkpoint_seconds": None,
            "wall_seconds": None,
            "end_to_end_seconds": None,
        }
        _save_summary(out, failed)
        return failed, None, out


def _write_bundle(config, run_id, baseline, baseline_dir, conditions,
                  summaries, run_dirs, pair_results):
    output_root = ROOT / config["experiment"].get("output_root", "outputs")
    bundle = output_root / "results" / "train_time_compaction_v1_1" / run_id
    bundle.mkdir(parents=True, exist_ok=False)
    summary_path = bundle / "summary.csv"
    fields = list(dict.fromkeys(field for summary in summaries for field in summary))
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    epoch_frames = []
    for experiment_id, run_dir in run_dirs.items():
        metrics_path = run_dir / "metrics.csv"
        rows = []
        if metrics_path.is_file() and metrics_path.stat().st_size:
            with metrics_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        for row in rows:
            row["source_experiment"] = experiment_id
        if rows:
            epoch_frames.extend(rows)
        destination = bundle / "runs" / experiment_id
        destination.mkdir(parents=True)
        for name in ("config.yaml", "environment.json", "run.log", "metrics.csv", "summary.json"):
            source_file = run_dir / name
            if source_file.is_file():
                shutil.copy2(source_file, destination / name)
        for mask_file in (run_dir / "masks").glob("*"):
            if mask_file.is_file():
                shutil.copy2(mask_file, destination / mask_file.name)

    baseline_destination = bundle / "baseline"
    baseline_destination.mkdir()
    for name in ("config.yaml", "environment.json", "run.log", "metrics.csv",
                 "summary.json", "mask_distances.csv"):
        source = baseline_dir / name
        if source.is_file():
            shutil.copy2(source, baseline_destination / name)
    for directory_name in ("masks", "plots"):
        source_dir = baseline_dir / directory_name
        if source_dir.is_dir():
            target_dir = baseline_destination / directory_name
            target_dir.mkdir()
            for source in source_dir.iterdir():
                if source.is_file():
                    shutil.copy2(source, target_dir / source.name)
    root_mask = baseline_dir / "stable_mask.pt"
    if root_mask.is_file():
        target_mask_dir = baseline_destination / "masks"
        target_mask_dir.mkdir(exist_ok=True)
        shutil.copy2(root_mask, target_mask_dir / root_mask.name)

    if epoch_frames:
        with (bundle / "metrics_all_epochs.csv").open("w", newline="", encoding="utf-8") as f:
            epoch_fields = list(dict.fromkeys(field for row in epoch_frames for field in row))
            writer = csv.DictWriter(f, fieldnames=epoch_fields)
            writer.writeheader()
            writer.writerows(epoch_frames)
    with (bundle / "compaction_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_results[0]))
        writer.writeheader()
        writer.writerows(pair_results)
    _json_write(bundle / "manifest.json", {
        "study": "Early-Bird Ver.1.1 train-time model compaction",
        "run_id": run_id,
        "baseline_experiment_id": baseline["experiment_id"],
        "tm": baseline["tm"],
        "strategies": config["comparison"]["strategies"],
        "information_modes": config["comparison"]["information_modes"],
        "conditions": conditions,
        "run_ids": [summary["experiment_id"] for summary in summaries],
        "failed_or_incomplete_runs": [
            summary["experiment_id"] for summary in summaries
            if summary.get("status") != "completed"],
        "checkpoint_policy": config["measurement"]["checkpoint_policy"],
        "large_checkpoints_copied": False,
    })
    (bundle / "README.md").write_text(
        "# Ver.1.1 train-time compaction comparison\n\n"
        f"Dense detector: `{baseline['experiment_id']}` (t_m={baseline['tm']}). "
        "Paired Dense-late-compaction and train-time-compaction runs are listed in "
        "`summary.csv`; epoch metrics are in `metrics_all_epochs.csv`. Failed runs are "
        "retained with their status and errors, and plots include completed runs only. "
        "Large checkpoints are kept in the project run folders and are not duplicated here.\n",
        encoding="utf-8")
    plot_compaction_comparison(summary_path, bundle / "plots")
    return bundle


def run_compaction_suite(config, resume_baseline=None):
    """Execute the full Ver.1.1 paired design and package its analysis artifacts."""
    measurement = config["measurement"]
    evaluation = measurement["evaluation"]
    checkpoint_policy = measurement["checkpoint_policy"]
    if int(measurement["validation_frequency_epochs"]) != 1:
        raise ValueError("Ver.1.1 currently requires validation every epoch")
    if int(measurement["epoch_log_frequency_epochs"]) != 1:
        raise ValueError("Ver.1.1 currently requires one log row per epoch")
    if not checkpoint_policy.get("initial") or not checkpoint_policy.get("final"):
        raise ValueError("Ver.1.1 requires initial and final checkpoints")
    if evaluation.get("checkpoint_format") != "torch_state_dict":
        raise ValueError("Only torch_state_dict checkpoints are implemented")
    if len(evaluation["input_shape"]) != 4 or int(evaluation["input_shape"][0]) != int(
            evaluation["inference_batch_size"]):
        raise ValueError("input_shape batch dimension must match inference_batch_size")
    if not torch.cuda.is_available() and config.get("runtime", {}).get("require_cuda", True):
        raise RuntimeError("CUDA-enabled PyTorch is required by the Ver.1.1 full-run config")
    seed = int(config["project"]["seed"])
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline, stable_masks, mask_source_id, run_id, baseline_dir = _load_baseline(
        config, resume_baseline, run_id, device)
    tm = int(baseline["tm"])
    dense_baseline_metrics = {
        "parameter_count": baseline.get("parameter_count"),
        "flops_per_sample": baseline.get("flops_per_sample"),
        "model_size_bytes": baseline.get("model_size_bytes"),
    }
    relative = config["ts"]["relative_to_tm"]
    conditions = {
        "C1_tm_minus_10": max(1, tm - int(relative["before_offset"])),
        "C2_tm": max(1, tm + int(relative["near_offset"])),
        "C3_tm_plus_10": max(1, tm + int(relative["after_offset"])),
        "C4_fixed_ts": int(config["ts"]["fixed_epoch"]),
    }
    modes = list(config["comparison"]["information_modes"])
    strategies = list(config["comparison"]["strategies"])
    if set(strategies) != {"dense_late_compaction", "train_time_compaction"}:
        raise ValueError("Ver.1.1 requires dense_late_compaction and train_time_compaction strategies")
    if set(modes) != {"online", "posthoc"}:
        raise ValueError("Ver.1.1 requires online and posthoc information modes")
    transition_checkpoint = bool(config["measurement"]["checkpoint_policy"].get("transition", False))

    summaries = []
    run_dirs = {}
    pair_results = []
    pair_counter = 0
    for mode in modes:
        for condition, ts in conditions.items():
            pair_counter += 1
            # Alternate arm order between condition pairs to reduce systematic
            # GPU thermal/order bias without changing either arm's seed.
            pair_strategies = strategies if pair_counter % 2 else list(reversed(strategies))
            pair = {}
            for strategy in pair_strategies:
                experiment_id = (
                    f"{run_id}_{condition}_{mode}_{strategy}_seed{seed}")
                summary, masks, out = _run_condition(
                    config, experiment_id, strategy, mode, ts, tm,
                    stable_masks, mask_source_id, transition_checkpoint,
                    dense_baseline_metrics)
                pair[strategy] = summary
                summaries.append(summary)
                run_dirs[experiment_id] = out

            control = pair["dense_late_compaction"]
            treatment = pair["train_time_compaction"]
            both_completed = (control.get("status") == "completed" and
                              treatment.get("status") == "completed")
            paired_mask_match = None
            if both_completed:
                control_mask = load_masks(
                    run_dirs[control["experiment_id"]] / "masks" / "transition_mask.pt")
                treatment_mask = load_masks(
                    run_dirs[treatment["experiment_id"]] / "masks" / "transition_mask.pt")
                paired_mask_match = _same_masks(control_mask, treatment_mask)
            for arm in (control, treatment):
                arm["paired_mask_match"] = paired_mask_match
                arm["paired_experiment_id"] = (
                    treatment["experiment_id"] if arm is control else control["experiment_id"])
                arm["speedup_after_ts"] = (
                    control.get("compact_train_seconds") / treatment.get("compact_train_seconds")
                    if both_completed and treatment.get("compact_train_seconds") else None)
                arm["total_speedup"] = (
                    control.get("end_to_end_seconds") / treatment.get("end_to_end_seconds")
                    if both_completed and treatment.get("end_to_end_seconds") else None)
                _save_summary(run_dirs[arm["experiment_id"]], arm)
                with (run_dirs[arm["experiment_id"]] / "run.log").open(
                        "a", encoding="utf-8") as log:
                    log.write(f"paired_mask_match={paired_mask_match}\n")
                if arm.get("status") == "completed":
                    _write_final_metrics_to_epoch_csv(
                        run_dirs[arm["experiment_id"]] / "metrics.csv", arm)
            pair_results.append({
                "condition": condition,
                "information_mode": mode,
                "ts": ts,
                "status": "completed" if both_completed else "incomplete",
                "paired_mask_match": paired_mask_match,
                "dense_control_id": control["experiment_id"],
                "train_time_compaction_id": treatment["experiment_id"],
                "dense_post_ts_train_seconds": control.get("compact_train_seconds"),
                "compact_post_ts_train_seconds": treatment.get("compact_train_seconds"),
                "compaction_seconds": treatment.get("compaction_seconds"),
                "dense_end_to_end_seconds": control.get("end_to_end_seconds"),
                "compact_end_to_end_seconds": treatment.get("end_to_end_seconds"),
                "speedup_after_ts": treatment["speedup_after_ts"],
                "total_speedup": treatment["total_speedup"],
                "dense_test_accuracy": control.get("accuracy"),
                "compact_test_accuracy": treatment.get("accuracy"),
                "dense_error": control.get("error"),
                "compact_error": treatment.get("error"),
            })

    summary_fields = list(dict.fromkeys(
        field for summary in summaries for field in summary))
    comparison_csv = ROOT / config["experiment"].get("output_root", "outputs") / \
        f"ts_compaction_comparison_{run_id}.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    run_summary = {
        "run_id": run_id,
        "baseline": baseline,
        "mask_source_run_id": mask_source_id,
        "conditions": summaries,
        "pairs": pair_results,
        "comparison_csv": str(comparison_csv),
    }
    _json_write(comparison_csv.with_name(f"run_summary_{run_id}.json"), run_summary)
    bundle = _write_bundle(config, run_id, baseline, baseline_dir, conditions,
                           summaries, run_dirs, pair_results)
    run_summary["bundle"] = str(bundle)
    return run_summary
