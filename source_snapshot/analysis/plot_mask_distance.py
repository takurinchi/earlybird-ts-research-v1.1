"""Plot recorded mask distances and optionally indicate detected t_m."""
import pandas as pd
import os
from pathlib import Path
MPL_CONFIG = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mask_distance(csv_path, output_path, tm=None):
    data = pd.read_csv(csv_path)
    if not {"epoch", "mask_distance"}.issubset(data.columns):
        raise ValueError("CSV must contain epoch and mask_distance")
    fig, ax = plt.subplots()
    ax.plot(data["epoch"], data["mask_distance"], marker=".")
    if tm is not None:
        ax.axvline(tm, color="red", linestyle="--", label=f"t_m={tm}")
        ax.legend()
    ax.set(xlabel="Epoch", ylabel="Mask distance")
    fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)
