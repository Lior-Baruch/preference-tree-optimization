"""Draw the README results figure from ``results/data/*.csv``.

Writes ``results/figures/results_light.png`` and ``results_dark.png`` (backgrounds match GitHub's
light and dark themes; the README picks one with a ``<picture>`` element).

Panel (a): the training reward Q1+Q2 by iteration for PTO with K=0 and K=5, mean +/- SE over the 96
personas, once per judge. The two judges use different absolute scales, so each gets its own y-axis.
Panel (b): base model (iteration 0) -> iteration 10 for the over-praise marker rate (judge-free
regex) and the MICI rate (MI-inconsistent behaviour) under both judges.

Needs only pandas + matplotlib:  python scripts/make_results_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"
FIGURES = ROOT / "results" / "figures"

JUDGES = {
    "gpt-4o-mini": "Training judge · gpt-4o-mini",
    "claude-haiku-4-5": "Held-out judge · claude-haiku-4-5",
}

# Series colours are validated for colour-vision deficiency against each surface.
THEMES = {
    "light": {
        "bg": "#ffffff", "ink": "#1f2328", "ink2": "#59636e", "muted": "#818b98",
        "grid": "#e8e8e4", "axis": "#c3c2b7", "k0": "#2a78d6", "k5": "#eb6834", "band": 0.14,
    },
    "dark": {
        "bg": "#0d1117", "ink": "#f0f6fc", "ink2": "#9198a1", "muted": "#7d8590",
        "grid": "#21262d", "axis": "#3d444d", "k0": "#3987e5", "k5": "#d95926", "band": 0.24,
    },
}
ARMS = (
    ("0", "k0", "PTO, K = 0 (no look-ahead)"),
    ("5", "k5", "PTO, K = 5 (5-utterance look-ahead)"),
)


def pick_font() -> str:
    """First installed UI sans from a short list (DejaVu Sans ships with matplotlib)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if name in installed:
            return name
    return "DejaVu Sans"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    kc = pd.read_csv(DATA / "k_contrast_by_iteration.csv")
    op = pd.read_csv(DATA / "overpraise_by_iteration.csv")
    return kc, op


def style_axis(ax: plt.Axes, t: dict) -> None:
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=8.5, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def draw_trajectories(ax: plt.Axes, kc: pd.DataFrame, judge: str, t: dict) -> None:
    d = kc[(kc["judge"] == judge) & (kc["instrument"] == "Q1+Q2")].sort_values("iteration")
    x = d["iteration"].to_numpy()
    for k, key, _ in ARMS:
        mean = d[f"mean_K{k}"].to_numpy()
        se = d[f"se_K{k}"].to_numpy()
        ax.fill_between(x, mean - se, mean + se, color=t[key], alpha=t["band"], linewidth=0)
        ax.plot(x, mean, color=t[key], linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter(x[-1:], mean[-1:], s=36, color=t[key], edgecolor=t["bg"], linewidth=1.5,
                   zorder=4)
    style_axis(ax, t)
    ax.set_xlim(-0.3, 10.3)
    ax.spines["bottom"].set_bounds(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("Iteration  (0 = base model)", color=t["ink2"], fontsize=8.5)
    ax.set_title(JUDGES[judge], loc="left", color=t["ink"], fontsize=10, pad=6)
    # Label the two iteration-10 endpoints, stacked so they never collide.
    end = d[d["iteration"] == 10].iloc[0]
    hi, lo = sorted((("0", end["mean_K0"]), ("5", end["mean_K5"])), key=lambda r: -r[1])
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.045
    for (k, v), dy in ((hi, pad), (lo, -pad)):
        ax.annotate(f"K={k}  {v:.3f}", (10, v), xytext=(10.25, v + dy), color=t["ink2"],
                    fontsize=8, va="center", ha="left", annotation_clip=False)


def draw_dumbbells(ax: plt.Axes, values: dict, title: str, t: dict, fmt: str) -> None:
    """One row per arm: open marker = iteration 0 (base model), filled marker = iteration 10."""
    for row, (k, key, _) in enumerate(ARMS):
        y = 1 - row
        v0, v10 = values[k]
        ax.plot([v0, v10], [y, y], color=t[key], linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([v0], [y], s=52, facecolor=t["bg"], edgecolor=t[key], linewidth=2, zorder=3)
        ax.scatter([v10], [y], s=52, color=t[key], edgecolor=t["bg"], linewidth=1.5, zorder=4)
        ax.annotate(fmt.format(v10), (v10, y), xytext=(7, 0), textcoords="offset points",
                    color=t["ink"], fontsize=8.5, va="center", ha="left")
    style_axis(ax, t)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_yticks([1, 0], [f"K = {k}" for k, _, _ in ARMS])
    ax.set_ylim(-0.55, 1.55)
    hi = max(max(v) for v in values.values())
    ax.set_xlim(-hi * 0.05, hi * 1.32)
    ax.set_title(title, loc="left", color=t["ink"], fontsize=9.5, pad=6)


def build(theme: str, kc: pd.DataFrame, op: pd.DataFrame) -> plt.Figure:
    t = THEMES[theme]
    plt.rcParams["font.family"] = pick_font()
    fig = plt.figure(figsize=(11.5, 7.4), facecolor=t["bg"])
    gs = fig.add_gridspec(2, 6, height_ratios=[1.55, 1.0], hspace=0.72, wspace=0.75,
                          left=0.06, right=0.95, top=0.86, bottom=0.08)

    fig.text(0.06, 0.965, "a", color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.08, 0.965, "Training reward Q1+Q2 (1–5 scale), mean ± SE over 96 personas",
             color=t["ink"], fontsize=11, va="top")
    handles = [Line2D([0], [0], color=t[key], linewidth=2.5, label=label)
               for _, key, label in ARMS]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.95, 0.985), ncol=2,
               frameon=False, fontsize=9, labelcolor=t["ink2"], handlelength=1.8)
    fig.text(0.08, 0.925,
             "The judges use different absolute scales — compare trends, not levels.",
             color=t["ink2"], fontsize=8.5, va="top")

    for col, judge in zip((slice(0, 3), slice(3, 6)), JUDGES, strict=True):
        draw_trajectories(fig.add_subplot(gs[0, col]), kc, judge, t)

    fig.text(0.06, 0.385, "b", color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.08, 0.385, "What look-ahead changes: base model (iteration 0, open) → "
             "iteration 10 (filled); lower is better", color=t["ink"], fontsize=11, va="top")

    lex = op.set_index(["k", "iteration"])["lexical_overpraise_rate"]
    panels = [(
        {k: (lex[(int(k), 0)], lex[(int(k), 10)]) for k, _, _ in ARMS},
        "Over-praise markers (regex, judge-free)\nshare of therapist turns", "{:.3f}",
    )]
    for judge, short in (("gpt-4o-mini", "training judge"), ("claude-haiku-4-5", "held-out judge")):
        m = kc[(kc["judge"] == judge) & (kc["instrument"] == "MICI")].set_index("iteration")
        panels.append((
            {k: (m.loc[0, f"mean_K{k}"], m.loc[10, f"mean_K{k}"]) for k, _, _ in ARMS},
            f"MI-inconsistent behaviour (MICI)\nper therapist turn · {short}", "{:.3f}",
        ))
    cols = (slice(0, 2), slice(2, 4), slice(4, 6))
    for col, (values, title, fmt) in zip(cols, panels, strict=True):
        draw_dumbbells(fig.add_subplot(gs[1, col]), values, title, t, fmt)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    kc, op = load()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        fig = build(theme, kc, op)
        out = args.out_dir / f"results_{theme}.png"
        fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
