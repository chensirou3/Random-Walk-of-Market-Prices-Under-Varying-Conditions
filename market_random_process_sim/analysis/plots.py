"""Plotting utilities for market random process experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_figure_dir(output_dir: str | Path) -> Path:
    """Create and return the figure output directory."""
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def ensure_formula_figure_dir(output_dir: str | Path) -> Path:
    """Create and return the formula figure output directory."""
    fig_dir = Path(output_dir) / "formula_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _color_cycle() -> list[str]:
    return [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#17becf",
    ]


def plot_price_path(
    prices: Sequence[float],
    output_dir: str | Path,
    filename: str,
    title: str = "Price Path",
    label: str = "Price",
) -> Path:
    """Plot a single price path."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(len(prices)), prices, label=label, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_return_series(
    returns: Sequence[float],
    output_dir: str | Path,
    filename: str,
    title: str = "Return Series",
) -> Path:
    """Plot daily log returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, len(returns) + 1), returns, label="Daily log return", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Log return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_return_distribution(
    returns: Sequence[float],
    output_dir: str | Path,
    filename: str,
    title: str = "Return Distribution",
    label: str = "Returns",
    bins: int = 40,
) -> Path:
    """Plot a histogram of daily log returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.asarray(returns, dtype=float), bins=bins, alpha=0.75, label=label, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Log return")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_multiple_price_paths(
    paths: Sequence[Sequence[float]] | Mapping[str, Sequence[float]],
    output_dir: str | Path,
    filename: str,
    title: str = "Multiple Price Paths",
    alpha: float = 0.45,
) -> Path:
    """Plot multiple price paths in one figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if isinstance(paths, Mapping):
        for label, prices in paths.items():
            ax.plot(np.arange(len(prices)), prices, label=label, linewidth=1.4, alpha=0.9)
        ax.legend()
    else:
        for i, prices in enumerate(paths):
            ax.plot(np.arange(len(prices)), prices, linewidth=0.9, alpha=alpha, label="Path" if i == 0 else None)
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_auction_curve(
    auction_curve: pd.DataFrame,
    output_dir: str | Path,
    filename: str,
    title: str = "Auction Demand and Supply Curve",
) -> Path:
    """Plot demand and supply curves from one auction."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(auction_curve["price"], auction_curve["demand"], label="Demand", linewidth=1.6)
    ax.plot(auction_curve["price"], auction_curve["supply"], label="Supply", linewidth=1.6)
    if "matched_volume" in auction_curve.columns and not auction_curve.empty:
        best_idx = int(auction_curve["matched_volume"].idxmax())
        ax.axvline(float(auction_curve.loc[best_idx, "price"]), color="black", linestyle="--", linewidth=1.0, label="Max volume price")
    ax.set_title(title)
    ax.set_xlabel("Candidate price")
    ax.set_ylabel("Quantity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_price_distribution_comparison(
    price_groups: Mapping[str, Sequence[float]],
    output_dir: str | Path,
    filename: str,
    title: str = "Price Distribution Comparison",
    bins: int = 40,
) -> Path:
    """Plot overlapping price distributions for multiple groups."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, prices in price_groups.items():
        ax.hist(np.asarray(prices, dtype=float), bins=bins, alpha=0.45, density=True, label=label)
    ax.set_title(title)
    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_deviation_series(
    deviation_groups: Mapping[str, Sequence[float]] | Sequence[float],
    output_dir: str | Path,
    filename: str,
    title: str = "Log Price Deviation",
) -> Path:
    """Plot log price deviations from the initial price."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if isinstance(deviation_groups, Mapping):
        for label, deviations in deviation_groups.items():
            ax.plot(np.arange(len(deviations)), deviations, label=label, linewidth=1.4)
    else:
        ax.plot(np.arange(len(deviation_groups)), deviation_groups, label="Log deviation", linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("log(P[t] / P[0])")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_mean_reversion_scatter(
    deviations: Sequence[float],
    future_returns: Sequence[float],
    beta: float,
    alpha: float,
    output_dir: str | Path,
    filename: str,
    title: str = "Mean Reversion Scatter",
) -> Path:
    """Plot future returns against current deviations with an OLS line."""
    x = np.asarray(deviations, dtype=float)
    y = np.asarray(future_returns, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, s=14, alpha=0.5, label="Daily observations")
    if x.size:
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(x_line, alpha + beta * x_line, color="black", linewidth=1.4, label="OLS fit")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Current log deviation")
    ax.set_ylabel("Next-day log return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_metric_vs_parameter(
    parameters: Sequence[float],
    metrics: Sequence[float],
    output_dir: str | Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
) -> Path:
    """Plot a metric against a scalar experiment parameter."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(parameters, metrics, marker="o", linewidth=1.6, label=ylabel)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_return_distribution_by_group(
    return_groups: Mapping[str, Sequence[float]],
    output_dir: str | Path,
    filename: str,
    title: str = "Return Distribution by Group",
    bins: int = 50,
) -> Path:
    """Plot return distributions for multiple groups."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, returns in return_groups.items():
        ax.hist(np.asarray(returns, dtype=float), bins=bins, histtype="step", density=True, linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_grouped_price_paths(
    path_groups: Mapping[str, Sequence[Sequence[float]]],
    output_dir: str | Path,
    filename: str,
    title: str,
    ylabel: str = "Price",
    alpha: float = 0.22,
    max_paths_per_group: int = 20,
    normalize_to_initial: bool = False,
) -> Path:
    """Plot multiple paths for several groups with one legend entry per group."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = _color_cycle()
    for group_idx, (label, paths) in enumerate(path_groups.items()):
        color = colors[group_idx % len(colors)]
        selected_paths = list(paths)[:max_paths_per_group]
        for path_idx, prices in enumerate(selected_paths):
            y = np.asarray(prices, dtype=float)
            if normalize_to_initial and y.size and y[0] != 0:
                y = y / y[0]
            ax.plot(
                np.arange(y.size),
                y,
                color=color,
                linewidth=0.9,
                alpha=alpha,
                label=label if path_idx == 0 else None,
            )
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_paths_with_mean(
    path_groups: Mapping[str, Sequence[Sequence[float]]],
    output_dir: str | Path,
    filename: str,
    title: str,
    ylabel: str = "Price",
    max_paths_per_group: int = 50,
    path_alpha: float = 0.18,
    group_colors: Mapping[str, str] | None = None,
) -> Path:
    """Plot multiple paths per group with one bold cross-path mean line per group."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    default_colors = _color_cycle()
    for group_idx, (label, paths) in enumerate(path_groups.items()):
        if group_colors and label in group_colors:
            color = group_colors[label]
        else:
            color = default_colors[group_idx % len(default_colors)]
        path_list = list(paths)
        if not path_list:
            continue
        selected = path_list[:max_paths_per_group]
        arr = np.asarray(selected, dtype=float)
        x = np.arange(arr.shape[1])
        for path_idx in range(arr.shape[0]):
            ax.plot(x, arr[path_idx], color=color, linewidth=0.8, alpha=path_alpha)
        full_arr = np.asarray(path_list, dtype=float)
        mean_path = full_arr.mean(axis=0)
        ax.plot(x, mean_path, color=color, linewidth=2.4, label=f"{label} (mean of {full_arr.shape[0]} paths)")
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_mean_paths_with_ci(
    path_groups: Mapping[str, Sequence[Sequence[float]]],
    output_dir: str | Path,
    filename: str,
    title: str,
    ylabel: str = "Price",
    group_colors: Mapping[str, str] | None = None,
) -> Path:
    """Plot only the cross-path mean line per group, with a 95% normal CI band."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    default_colors = _color_cycle()
    for group_idx, (label, paths) in enumerate(path_groups.items()):
        if group_colors and label in group_colors:
            color = group_colors[label]
        else:
            color = default_colors[group_idx % len(default_colors)]
        path_list = list(paths)
        if not path_list:
            continue
        arr = np.asarray(path_list, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 1:
            continue
        x = np.arange(arr.shape[1])
        mean_path = arr.mean(axis=0)
        std_path = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_path)
        sem = std_path / np.sqrt(max(arr.shape[0], 1))
        lower = mean_path - 1.96 * sem
        upper = mean_path + 1.96 * sem
        ax.fill_between(x, lower, upper, color=color, alpha=0.18)
        ax.plot(x, mean_path, color=color, linewidth=2.2, label=f"{label} (mean of {arr.shape[0]} paths)")
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_boxplot_by_group(
    values_by_group: Mapping[str, Sequence[float]],
    output_dir: str | Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    log_x_labels: bool = False,
) -> Path:
    """Plot grouped distributions as a boxplot with mean markers."""
    labels = list(values_by_group.keys())
    values = [np.asarray(values_by_group[label], dtype=float) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(values, labels=labels, showmeans=True, meanline=True)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_x_labels:
        ax.text(
            0.99,
            0.02,
            "x-axis labels are wealth multipliers",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="dimgray",
        )
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_metric_with_ci(
    parameters: Sequence[float],
    means: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    output_dir: str | Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
) -> Path:
    """Plot group means with 95% confidence intervals."""
    x = np.asarray(parameters, dtype=float)
    y = np.asarray(means, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    yerr = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=1.6, label="Mean with 95% CI")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_table_image(
    table: pd.DataFrame,
    output_dir: str | Path,
    filename: str,
    title: str,
    max_rows: int = 12,
) -> Path:
    """Render a compact DataFrame as a report-ready PNG table."""
    display = table.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: f"{x:.6g}")

    fig_height = max(2.5, 0.38 * (len(display) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title(title, pad=12)
    table_artist = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1, 1.25)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_multi_line_with_ci(
    data: pd.DataFrame,
    group_col: str,
    x_col: str,
    y_col: str,
    lower_col: str | None,
    upper_col: str | None,
    output_dir: str | Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
) -> Path:
    """Plot one line per group with optional confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = _color_cycle()
    for idx, (group, group_df) in enumerate(data.groupby(group_col, sort=False)):
        group_df = group_df.sort_values(x_col)
        color = colors[idx % len(colors)]
        x = group_df[x_col].to_numpy(dtype=float)
        y = group_df[y_col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.6, color=color, label=str(group))
        if lower_col and upper_col:
            lo = group_df[lower_col].to_numpy(dtype=float)
            hi = group_df[upper_col].to_numpy(dtype=float)
            ax.fill_between(x, lo, hi, color=color, alpha=0.16)
    if log_x:
        ax.set_xscale("log")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_scatter_with_fit(
    x: Sequence[float],
    y: Sequence[float],
    output_dir: str | Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    label: str = "Runs",
    log_x: bool = False,
) -> Path:
    """Plot a scatter with an OLS fit line."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_arr, y_arr, s=24, alpha=0.55, label=label)
    if x_arr.size >= 2 and np.var(x_arr) > 1e-15:
        alpha, beta = np.linalg.lstsq(np.column_stack([np.ones_like(x_arr), x_arr]), y_arr, rcond=None)[0]
        x_line = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), 100)
        ax.plot(x_line, alpha + beta * x_line, color="black", linewidth=1.4, label="OLS fit")
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_grouped_bar(
    table: pd.DataFrame,
    category_col: str,
    value_cols: Sequence[str],
    output_dir: str | Path,
    filename: str,
    title: str,
    ylabel: str,
) -> Path:
    """Plot selected normalized metrics as grouped bars."""
    categories = table[category_col].astype(str).to_list()
    values = table[list(value_cols)].astype(float).copy()
    for col in values.columns:
        scale = float(np.nanmax(np.abs(values[col].to_numpy(dtype=float))))
        if scale > 1e-15:
            values[col] = values[col] / scale

    x = np.arange(len(categories))
    width = 0.8 / max(len(value_cols), 1)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for idx, col in enumerate(value_cols):
        ax.bar(x + idx * width - 0.4 + width / 2, values[col], width=width, label=col)
    ax.set_title(title)
    ax.set_xlabel("Market type")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_dual_axis_chain(
    x: Sequence[float],
    y1: Sequence[float],
    y2: Sequence[float],
    output_dir: str | Path,
    filename: str,
    title: str,
    xlabel: str,
    y1label: str,
    y2label: str,
    log_x: bool = False,
) -> Path:
    """Plot two mechanism-chain metrics against a common x-axis."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(x, y1, marker="o", color="#1f77b4", linewidth=1.6, label=y1label)
    ax2.plot(x, y2, marker="s", color="#d62728", linewidth=1.6, label=y2label)
    if log_x:
        ax1.set_xscale("log")
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color="#1f77b4")
    ax2.set_ylabel(y2label, color="#d62728")
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    return _save(fig, ensure_figure_dir(output_dir) / filename)


def plot_panel_price_paths(
    left_paths: Sequence[Sequence[float]],
    right_paths: Sequence[Sequence[float]],
    output_dir: str | Path,
    filename: str,
    title: str,
    left_title: str,
    right_title: str,
    normalize_to_initial: bool = False,
    max_paths: int = 25,
) -> Path:
    """Plot two panels of multiple price paths."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, paths, subtitle, color in [
        (axes[0], left_paths, left_title, "#1f77b4"),
        (axes[1], right_paths, right_title, "#d62728"),
    ]:
        for path in list(paths)[:max_paths]:
            y = np.asarray(path, dtype=float)
            if normalize_to_initial and y.size and y[0] != 0:
                y = y / y[0]
            ax.plot(np.arange(y.size), y, color=color, alpha=0.35, linewidth=0.9)
        ax.set_title(subtitle)
        ax.set_xlabel("Day")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Price / initial price" if normalize_to_initial else "Price")
    fig.suptitle(title)
    return _save(fig, ensure_figure_dir(output_dir) / filename)
