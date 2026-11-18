"""Plot a selected outcome against t_s, keeping information modes separate."""
import pandas as pd
import os
from pathlib import Path
MPL_CONFIG = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ts_comparison(csv_path, output_path, metric="accuracy"):
    data = pd.read_csv(csv_path)
    required = {"ts", metric}
    if not required.issubset(data.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    group = "information_mode" if "information_mode" in data else None
    fig, ax = plt.subplots()
    for name, frame in (data.groupby(group) if group else [(metric, data)]):
        ax.plot(frame["ts"], frame[metric], marker="o", label=str(name))
    ax.set(xlabel="t_s", ylabel=metric)
    if group:
        ax.legend(title=group)
    fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)
