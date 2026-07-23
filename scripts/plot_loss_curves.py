#!/usr/bin/env python3
"""Plot TASK-former training losses from train.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch-output", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=15)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    expected = {
        "epoch",
        "step",
        "loss",
        "embedding",
        "classification",
        "decoder",
    }
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    epoch_means = (
        frame.groupby("epoch", as_index=False)[
            ["loss", "embedding", "classification", "decoder"]
        ]
        .mean()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    epoch_means.to_csv(args.epoch_output, index=False, float_format="%.6f")

    smooth = frame[
        ["loss", "embedding", "classification", "decoder"]
    ].rolling(args.smooth_window, center=True, min_periods=1).mean()
    boundaries = frame.groupby("epoch")["step"].min().iloc[1:].tolist()

    colors = {
        "embedding": "#0072B2",
        "classification": "#E69F00",
        "decoder": "#009E73",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(frame["step"], frame["loss"], color="#D55E00", alpha=0.16, linewidth=0.7, label="Raw")
    ax.plot(frame["step"], smooth["loss"], color="#D55E00", linewidth=2.0, label="Smoothed")
    ax.set_title("Total training loss")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Weighted loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for name, color in colors.items():
        ax.plot(frame["step"], smooth[name], color=color, linewidth=1.7, label=name.capitalize())
    ax.set_title("Loss components (smoothed)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Unweighted component loss")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(epoch_means["epoch"], epoch_means["loss"], color="#D55E00", marker="o", linewidth=2.0)
    ax.set_title("Mean total loss by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean weighted loss")
    ax.set_xticks(epoch_means["epoch"])

    ax = axes[1, 1]
    for name, color in colors.items():
        ax.plot(epoch_means["epoch"], epoch_means[name], color=color, marker="o", linewidth=1.7, label=name.capitalize())
    ax.set_title("Mean component losses by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean unweighted component loss")
    ax.set_xticks(epoch_means["epoch"])
    ax.legend(frameon=False)

    for ax in axes[0]:
        for boundary in boundaries:
            ax.axvline(boundary, color="black", alpha=0.12, linewidth=0.8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"TASK-former 10-epoch loss curves (rolling mean: {args.smooth_window} logged points)",
        fontsize=15,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
