"""Plan or execute the CIFAR-10 Early-Bird and t_s experiment suite."""
import argparse
import json
from pathlib import Path
from cifar_experiment import run_full
from cifar_compaction_experiment import run_compaction_suite
from utils.config import load_config, require_config


def build_plan(config_path):
    cfg = load_config(config_path)
    required = ["project.seed", "experiment.epochs", "experiment.dataset",
        "experiment.model.factory", "data.loader_factory", "pruning.selected_ratio",
        "earlybird.epsilon", "earlybird.mask_window", "ts.relative_to_tm.before_offset",
        "ts.relative_to_tm.after_offset", "ts.fixed_epoch"]
    mode = cfg["experiment"].get("mode", "ts_comparison")
    if mode == "train_time_compaction":
        required.extend(["comparison.strategies", "comparison.information_modes",
                         "compaction.optimizer_state", "compaction.scheduler",
                         "measurement.validation_frequency_epochs",
                         "measurement.epoch_log_frequency_epochs",
                         "measurement.checkpoint_policy",
                         "measurement.evaluation.labels",
                         "measurement.evaluation.input_shape",
                         "measurement.evaluation.inference_batch_size",
                         "measurement.evaluation.warmup",
                         "measurement.evaluation.repeats",
                         "measurement.evaluation.checkpoint_format"])
    else:
        required.append("ts.information_modes")
    require_config(cfg, required)
    tm = "detected from Early-Bird search run"
    plan = {"status": "implementation_present_validation_pending",
        "mode": mode,
        "dataset": cfg["experiment"]["dataset"],
        "model": "CIFAR VGG16-BN", "epochs": cfg["experiment"]["epochs"],
        "seed": cfg["project"]["seed"], "ratio": cfg["pruning"]["selected_ratio"],
        "detection": {"epsilon": cfg["earlybird"]["epsilon"],
                      "mask_window": cfg["earlybird"]["mask_window"]},
        "ts": tm,
        "information_modes": (cfg["comparison"]["information_modes"]
                              if mode == "train_time_compaction"
                              else cfg["ts"]["information_modes"])}
    if plan["mode"] == "train_time_compaction":
        plan["strategies"] = cfg["comparison"]["strategies"]
        plan["information_modes"] = cfg["comparison"]["information_modes"]
        plan["conditions"] = {
            "C1": f"t_m - {cfg['ts']['relative_to_tm']['before_offset']}",
            "C2": f"t_m + {cfg['ts']['relative_to_tm']['near_offset']}",
            "C3": f"t_m + {cfg['ts']['relative_to_tm']['after_offset']}",
            "C4": cfg["ts"]["fixed_epoch"],
        }
        plan["optimizer_state_policy"] = cfg["compaction"]["optimizer_state"]
        plan["scheduler_policy"] = cfg["compaction"]["scheduler"]
        plan["validation_frequency_epochs"] = cfg["measurement"]["validation_frequency_epochs"]
        plan["checkpoint_policy"] = cfg["measurement"]["checkpoint_policy"]
    else:
        plan["status"] = "ready_to_execute"
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ts_sweep.yaml")
    parser.add_argument("--execute", action="store_true", help="run full training and t_s comparisons")
    parser.add_argument("--resume-baseline", help="resume comparisons from a completed baseline run folder")
    parser.add_argument("--output")
    args = parser.parse_args()
    plan = build_plan(args.config)
    if not args.execute:
        output = args.output or ("experiment_plan_v1_1.json"
                                 if plan["mode"] == "train_time_compaction"
                                 else "experiment_plan.json")
        Path(output).write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(json.dumps(plan, indent=2))
        return
    config = load_config(args.config)
    if config.get("experiment", {}).get("mode") == "train_time_compaction":
        results = run_compaction_suite(config, resume_baseline=args.resume_baseline)
    else:
        results = run_full(config, resume_baseline=args.resume_baseline)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
