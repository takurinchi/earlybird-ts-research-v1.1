"""Validate experiment configuration and prepare a reproducible run directory."""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml
from utils.config import load_config, require_config


def prepare_run(config_path, experiment_id=None):
    cfg_path = Path(config_path).resolve()
    config = load_config(cfg_path)
    required = ["project.seed", "experiment.epochs", "experiment.model.factory",
                "data.loader_factory"]
    mode = config.get("experiment", {}).get("mode", "dense_baseline")
    if mode in ("earlybird_detection", "ts_comparison"):
        required.append("pruning.selected_ratio")
    if mode == "ts_comparison":
        required.extend(["experiment.dataset", "ts.relative_to_tm.before_offset",
                         "ts.relative_to_tm.after_offset", "ts.fixed_epoch",
                         "ts.information_mode"])
    require_config(config, required)
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(config["experiment"].get("output_root", "outputs")) / experiment_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "checkpoints").mkdir(); (out / "masks").mkdir(); (out / "plots").mkdir()
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    env = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch
        env.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda})
    except ImportError:
        env["torch"] = None
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (out / "run.log").touch(); (out / "metrics.csv").touch(); (out / "mask_distances.csv").touch()
    (out / "summary.json").write_text(json.dumps({"experiment_id": experiment_id,
        "status": "prepared_not_run", "tm_detected": False, "tm": None,
        "ts": config["experiment"].get("ts"), "warnings": ["Training adapter not configured"]},
        indent=2), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    print(prepare_run(args.config, args.experiment_id))


if __name__ == "__main__":
    main()
