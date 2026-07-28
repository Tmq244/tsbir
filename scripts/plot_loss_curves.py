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
    parser.add_argument("--title")
    parser.add_argument(
        "--exclude-plot-components",
        nargs="+",
        default=(),
        help="Loss components to retain in the CSV summary but omit from the plot.",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"epoch", "step", "loss"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    component_order = [
        "embedding",
        "classification",
        "decoder",
        "consistency",
        "retrieval",
        "reliability",
    ]
    summary_components = [name for name in component_order if name in frame.columns]
    if not summary_components:
        raise ValueError("no recognized loss-component columns found")
    unknown_exclusions = set(args.exclude_plot_components).difference(summary_components)
    if unknown_exclusions:
        raise ValueError(
            f"cannot exclude unavailable components: {sorted(unknown_exclusions)}"
        )
    plot_components = [
        name
        for name in summary_components
        if name not in args.exclude_plot_components
    ]
    if not plot_components:
        raise ValueError("all recognized loss components were excluded from the plot")

    loss_columns = ["loss", *summary_components]

    epoch_means = (
        frame.groupby("epoch", as_index=False)[loss_columns]
        .mean()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.epoch_output.parent.mkdir(parents=True, exist_ok=True)
    epoch_means.to_csv(args.epoch_output, index=False, float_format="%.6f")

    smooth = frame[loss_columns].rolling(
        args.smooth_window,
        center=True,
        min_periods=1,
    ).mean()
    boundaries = frame.groupby("epoch")["step"].min().iloc[1:].tolist()

    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
    colors = dict(zip(plot_components, palette))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(frame["step"], frame["loss"], color="#D55E00", alpha=0.16, linewidth=0.7, label="Raw")
    ax.plot(frame["step"], smooth["loss"], color="#D55E00", linewidth=2.0, label="Smoothed")
    ax.set_title("Total training loss")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Weighted loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for name in plot_components:
        ax.plot(
            frame["step"],
            smooth[name],
            color=colors[name],
            linewidth=1.7,
            label=name.replace("_", " ").title(),
        )
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
    for name in plot_components:
        ax.plot(
            epoch_means["epoch"],
            epoch_means[name],
            color=colors[name],
            marker="o",
            linewidth=1.7,
            label=name.replace("_", " ").title(),
        )
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

    title = args.title or args.input.parent.name
    epoch_count = frame["epoch"].nunique()
    fig.suptitle(
        f"{title}: {epoch_count}-epoch loss curves "
        f"(rolling mean: {args.smooth_window} logged points)",
        fontsize=15,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
