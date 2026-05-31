"""Shared plotting style for COSMOS paper-convergence figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

BANDS = ("u", "g", "r", "i", "z", "y")

COLORS = {
    "galaxy": "#d62728",
    "star": "#1f77b4",
    "dp2_only": "#7f7f7f",
    "external_only": "#ff7f0e",
    "matched": "#111111",
    "threshold": "#222222",
}

FIG_SIZES = {
    "1x3": (17.2, 5.4),
    "2x2": (12.4, 10.2),
    "2x4": (19.2, 11.0),
    "3x3": (15.4, 13.2),
    "3x2": (12.4, 13.0),
    "2x3": (14.2, 8.8),
}

COLOR_COLOR_LIMITS = {
    ("ug", "gr"): ((-0.5, 4.0), (-0.5, 2.5)),
    ("gr", "ri"): ((-0.5, 2.0), (-0.5, 2.5)),
    ("ri", "iz"): ((-0.5, 2.5), (-0.5, 1.3)),
    ("iz", "zy"): ((-0.5, 1.3), (-0.7, 1.3)),
}

DEFAULT_DPI = 200


def set_paper_style() -> None:
    """Apply a consistent, quiet plotting style."""

    plt.rcParams.update(
        {
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.titlepad": 8,
            "axes.labelpad": 6,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "axes.axisbelow": True,
        }
    )


def save_figure(fig, output_png: Path | str, write_pdf: bool = True) -> list[Path]:
    """Save a figure as PNG and optionally as a same-stem PDF."""

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    fig.savefig(output_png, bbox_inches="tight", pad_inches=0.08)
    saved.append(output_png)
    if write_pdf:
        output_pdf = output_png.with_suffix(".pdf")
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.08)
        saved.append(output_pdf)
    plt.close(fig)
    return saved


def downsample_frame(df, max_points: int, random_state: int = 42):
    """Return a reproducible plotting subset without changing counts elsewhere."""

    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=random_state)
