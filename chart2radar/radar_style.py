from __future__ import annotations

from textwrap import wrap
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


FONT_FAMILY = "Times New Roman"

# Pure white canvas with muted neutral gridlines
PAPER_BG_COLOR = "#FFFFFF"
PLOT_BG_COLOR = "#FFFFFF"
GRID_COLOR_MAJOR = mpl.colors.to_rgba("#CBD1D6", 0.85)
GRID_COLOR_MINOR = mpl.colors.to_rgba("#D9DEE3", 0.45)

mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.serif": [FONT_FAMILY],
        "figure.facecolor": PAPER_BG_COLOR,
        "axes.facecolor": PLOT_BG_COLOR,
        "savefig.facecolor": PAPER_BG_COLOR,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 11.5,
        "figure.autolayout": False,
        "text.color": "#000000",
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
    }
)

"""Legend and color helpers for consistent plotting."""


# Harmonised palette matching the provided reference style
LINE_COLORS: Dict[str, str] = {
    # Baselines / families mapped onto the 4-colour scheme
    "GPT-5": "#F28E8C",  # UNI coral
    "GPT-o4-mini": "#56B3C4",  # REMEDIS teal
    "gpt-o4": "#56B3C4",
    "ChatGPT-4o": "#56B3C4",
    "DeepSeek-R1": "#E6C24F",  # CTransPath sunflower
    "DeepSeek": "#E6C24F",
    "Qwen3-235b": "#B88FD6",  # ResNet-50 lilac
    "Qwen3": "#B88FD6",
    # OSS ladders inherit softened variants
    "gpt-oss-20b": "#F6A69F",
    "gpt-oss-20b (L)": "#F9B8B2",
    "gpt-oss-20b (M)": "#F49389",
    "gpt-oss-20b (H)": "#ED7C72",
    "gpt-oss-20b finetuned": "#A7D8A6",  # Green for finetuned
    "OSS-20b (L)": "#F9B8B2",
    "OSS-20b (M)": "#F49389",
    "OSS-20b (H)": "#ED7C72",
    "gpt-oss-120b": "#C7A3E1",
    "gpt-oss-120b (L)": "#D6B8E9",
    "gpt-oss-120b (M)": "#C59BDD",
    "gpt-oss-120b (H)": "#B280D1",
    "OSS-120b (L)": "#D6B8E9",
    "OSS-120b (M)": "#C59BDD",
    "OSS-120b (H)": "#B280D1",
}


def color_pair(name: str) -> Tuple[str, Tuple[float, float, float, float]]:
    """Return a publication-friendly line/fill colour pairing."""

    line = LINE_COLORS.get(name, "#7F7F7F")
    rgba = mpl.colors.to_rgba(line, 0.22)
    return line, rgba


def legend_label(name: str) -> str:
    """Normalise model names to standard legend labels, preserving L/M/H."""

    key = name.lower()
    variant = None
    if "(l)" in key:
        variant = " (L)"
    elif "(m)" in key:
        variant = " (M)"
    elif "(h)" in key:
        variant = " (H)"

    if "gpt-5" in key:
        return "GPT-5"
    if "chatgpt-4o" in key or "chatgpt4o" in key or "o4" in key:
        return "GPT-o4-mini"
    if "deepseek" in key:
        return "DeepSeek-R1"
    if "qwen3" in key:
        return "Qwen3-235b"
    if "finetuned" in key or "13beams" in key:
        return "gpt-oss-20b finetuned"
    if "oss-20b" in key:
        return f"gpt-oss-20b{variant or ''}"
    if "oss-120b" in key:
        return f"gpt-oss-120b{variant or ''}"
    return name


def format_section_labels(
    labels: Iterable[str],
    line_length: int = 26,
    line_break: str = "\n",
) -> List[str]:
    """Wrap long section names for polar axes."""

    formatted: List[str] = []
    for label in labels:
        if len(label) <= line_length:
            formatted.append(label)
            continue
        wrapped = wrap(label, width=line_length, break_long_words=False, drop_whitespace=False)
        formatted.append(line_break.join(wrapped))
    return formatted


def make_base_axes(
    theta_labels: Sequence[str],
    radial_tickvals: Sequence[int] | None = None,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray]:
    """Create a polar axis tuned for camera-ready output."""

    num_axes = len(theta_labels)
    if num_axes < 3:
        raise ValueError("Radar plots require at least three categories.")

    angles = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)

    fig = plt.figure(figsize=(12.5, 12.5))
    fig.patch.set_facecolor(PAPER_BG_COLOR)

    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(PLOT_BG_COLOR)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles)
    ax.set_xticklabels([""] * num_axes)
    ax.tick_params(axis="x", pad=0, labelsize=0)

    tickvals = list(radial_tickvals or [0, 20, 40, 60, 80, 100])
    max_tick = max(tickvals)
    ax.set_ylim(0, max_tick)
    ax.set_yticks(tickvals)
    ax.set_yticklabels(
        [f"{tick:g}" for tick in tickvals],
        color="#000000",
        ha="center",
        va="center",
    )
    ax.tick_params(axis="y", labelsize=11.5, colors="#000000", pad=8)
    ax.set_rlabel_position(90)
    for label in ax.get_yticklabels():
        label.set_fontweight("regular")
        label.set_bbox(None)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID_COLOR_MAJOR, linewidth=1.1, linestyle="-")
    ax.grid(True, which="minor", color=GRID_COLOR_MINOR, linewidth=0.7, linestyle="-")
    ax.spines["polar"].set_edgecolor("#C0C7CE")
    ax.spines["polar"].set_linewidth(1.1)

    return fig, ax, angles


def add_angular_labels(*args, **kwargs) -> None:
    # Legacy no-op preserved for compatibility with earlier scripts
    return None


def place_section_labels(ax: plt.Axes, angles: np.ndarray, labels: Sequence[str]) -> None:
    """Position section labels with angle-aware padding to avoid polygon overlap."""

    transform = ax.get_xaxis_transform()
    base_offset = 0.97
    horizontal_boost = 0.08
    vertical_boost = 0.06

    for angle, label in zip(angles, labels):
        # Due to theta offset, sin(angle) maps to Cartesian x, cos(angle) to y
        x_component = np.sin(angle)
        y_component = np.cos(angle)

        radial = base_offset + horizontal_boost * abs(x_component) + vertical_boost * abs(y_component)

        if x_component > 0.3:
            h_align = "left"
        elif x_component < -0.3:
            h_align = "right"
        else:
            h_align = "center"

        if y_component > 0.4:
            v_align = "bottom"
        elif y_component < -0.4:
            v_align = "top"
        else:
            v_align = "center"

        text = ax.text(
            angle,
            radial,
            label,
            ha=h_align,
            va=v_align,
            color="#000000",
            fontsize=13,
            bbox=dict(
                facecolor="#FFFFFF",
                edgecolor="#B8C3CE",
                linewidth=1.5,
                boxstyle="round,pad=0.35",
            ),
            zorder=10,
            transform=transform,
        )
        text.set_clip_on(False)
