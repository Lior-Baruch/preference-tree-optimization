"""Draw the README "earlier experiments" figure from ``results/data``.

Writes ``results/figures/history_light.png`` and ``history_dark.png`` (backgrounds match GitHub's
light and dark themes; the README picks one with a ``<picture>`` element).

Each panel shows the evaluation score on the instrument the arm was trained on, by iteration, for
PTO with K = 0 and K = 5 (mean +/- SE over the 96 personas; iteration 0 is the untrained base
model, dashed line):

- Generation 2 (three reward oracles): one panel per oracle, each on its own instrument,
  ``gen2_by_state.csv``. The dagger marks the Q1+Q2-reward K = 5 iteration-6 evaluation, which
  was partly regenerated in a technical re-run and is not used in any number.
- Generation 3 (this repository): Q1+Q2 under the training judge,
  ``k_contrast_by_iteration.csv``.

Every panel has its own y-axis: therapist, patient and judge differ between generations, so
absolute scores are not comparable across generations.

Needs only pandas + matplotlib:  python scripts/make_history_figure.py
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

# Same tokens as scripts/make_results_figure.py (series colours validated for colour-vision
# deficiency against each surface).
THEMES = {
    "light": {
        "bg": "#ffffff", "ink": "#1f2328", "ink2": "#59636e", "muted": "#818b98",
        "grid": "#e8e8e4", "axis": "#c3c2b7", "k0": "#2a78d6", "k5": "#eb6834", "band": 0.14,
        "base": "#818b98",
    },
    "dark": {
        "bg": "#0d1117", "ink": "#f0f6fc", "ink2": "#9198a1", "muted": "#7d8590",
        "grid": "#21262d", "axis": "#3d444d", "k0": "#3987e5", "k5": "#d95926", "band": 0.18,
        "base": "#7d8590",
    },
}
ARMS = (("0", "k0", "PTO, K = 0 (no look-ahead)"), ("5", "k5", "PTO, K = 5 (look-ahead)"))

GEN2_PANELS = (  # (oracle, instrument column, panel title, scale note)
    ("Q1Q2", "Q1+Q2", "Generation 2 · Q1+Q2 reward", "scored on Q1+Q2 (1–5)"),
    ("WAI", "WAI_SR", "Generation 2 · WAI-SR reward", "scored on WAI-SR (1–5)"),
    ("CSQ8", "CSQ8", "Generation 2 · CSQ-8 reward", "scored on CSQ-8 (1–4)"),
)


def pick_font() -> str:
    """First installed UI sans from a short list (DejaVu Sans ships with matplotlib)."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if name in installed:
            return name
    return "DejaVu Sans"


def style_axis(ax: plt.Axes, t: dict) -> None:
    ax.set_facecolor(t["bg"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def load() -> dict[str, pd.DataFrame]:
    """Long frames with columns K ("0"/"5"), iteration, mean, se — one per panel."""
    panels: dict[str, pd.DataFrame] = {}

    g2 = pd.read_csv(DATA / "gen2_by_state.csv", dtype={"K": str}, keep_default_na=False)
    for oracle, instrument, _, _ in GEN2_PANELS:
        d = g2[g2["instrument"] == instrument]
        base = d[d["state"] == "Base"]
        arm = d[d["oracle"] == oracle]
        panels[f"gen2_{oracle}"] = pd.concat(
            [pd.concat([base.assign(K=k), arm[arm["K"] == k]]) for k, _, _ in ARMS]
        )[["K", "iteration", "mean", "se"]]

    kc = pd.read_csv(DATA / "k_contrast_by_iteration.csv")
    kc = kc[(kc["judge"] == "gpt-4o-mini") & (kc["instrument"] == "Q1+Q2")]
    panels["gen3"] = pd.concat(
        [kc.assign(K=k, mean=kc[f"mean_K{k}"], se=kc[f"se_K{k}"]) for k, _, _ in ARMS]
    )[["K", "iteration", "mean", "se"]]
    return panels


def draw_panel(ax: plt.Axes, d: pd.DataFrame, title: str, note: str, t: dict, *,
               base_level: float, xmax: int, xstep: int) -> None:
    ax.axhline(base_level, color=t["base"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    for k, key, _ in ARMS:
        s = d[d["K"] == k].sort_values("iteration")
        x, m, se = s["iteration"].to_numpy(), s["mean"].to_numpy(), s["se"].to_numpy()
        ax.fill_between(x, m - se, m + se, color=t[key], alpha=t["band"], linewidth=0)
        ax.plot(x, m, color=t[key], linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter(x[-1:], m[-1:], s=30, color=t[key], edgecolor=t["bg"], linewidth=1.5,
                   zorder=4)
    style_axis(ax, t)
    ax.set_xlim(-0.35, xmax + 0.35)
    ax.spines["bottom"].set_bounds(0, xmax)
    ax.set_xticks(range(0, xmax + 1, xstep))
    ax.set_xlabel("Iteration", color=t["ink2"], fontsize=9)
    ax.set_title(note, loc="left", color=t["ink2"], fontsize=9, pad=6)
    ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 22),
                textcoords="offset points", color=t["ink"], fontsize=10.5, fontweight="bold",
                ha="left", va="bottom")


def build(theme: str, panels: dict[str, pd.DataFrame]) -> plt.Figure:
    t = THEMES[theme]
    plt.rcParams["font.family"] = pick_font()
    fig = plt.figure(figsize=(11.5, 4.9), facecolor=t["bg"])
    outer = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.05], wspace=0.13,
                             left=0.05, right=0.985, top=0.60, bottom=0.17)
    inner = outer[0, 0].subgridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.36)

    fig.text(0.05, 0.975, "Evaluation score on the training reward, by iteration",
             color=t["ink"], fontsize=12, va="top")
    fig.text(0.05, 0.922, "Mean ± SE over 96 personas; iteration 0 is the untrained base model "
             "(dashed). Each generation has its own therapist, patient and judge: compare "
             "within a generation, never across.", color=t["ink2"], fontsize=9, va="top")
    handles = [Line2D([0], [0], color=t[key], linewidth=2.5, label=label)
               for _, key, label in ARMS]
    handles.append(Line2D([0], [0], color=t["base"], linewidth=1.2, linestyle=(0, (4, 3)),
                          label="untrained base"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.88), ncol=3,
               frameon=False, fontsize=9, labelcolor=t["ink2"], handlelength=1.8,
               columnspacing=1.6)
    fig.text(0.05, 0.025, "† Generation 2, Q1+Q2 reward, K = 5, iteration 6: partly regenerated "
             "in a technical re-run; not used in any number.", color=t["ink2"], fontsize=8.5,
             va="bottom")

    def setup(spec, text: str) -> None:
        fig.text(spec.get_position(fig).x0, 0.765, text, color=t["ink2"], fontsize=8.5,
                 va="top")

    # Generation 2 -------------------------------------------------------------------------
    setup(outer[0, 0], "Llama-3.2-1B (4-bit) · gpt-4o-mini patient and judge · "
          "each arm scored on its own training reward")
    for col, (oracle, _, title, note) in enumerate(GEN2_PANELS):
        ax = fig.add_subplot(inner[0, col])
        d = panels[f"gen2_{oracle}"]
        xmax = int(d["iteration"].max())
        draw_panel(ax, d, title, note, t, base_level=d[d["iteration"] == 0]["mean"].iloc[0],
                   xmax=xmax, xstep=2 if xmax > 6 else 1)
        if oracle == "Q1Q2":
            v6 = d[(d["K"] == "5") & (d["iteration"] == 6)].iloc[0]
            ax.annotate("†", (6, v6["mean"]), xytext=(0, -9), textcoords="offset points",
                        ha="center", va="top", fontsize=10, color=t["ink"])

    # Generation 3 -------------------------------------------------------------------------
    setup(outer[0, 1], "Llama-3.2-1B (bf16) · gpt-4o-mini")
    ax = fig.add_subplot(outer[0, 1])
    g3 = panels["gen3"]
    # each arm drew its own iteration-0 sample; the dashed line is their average
    base3 = g3[g3["iteration"] == 0]["mean"].mean()
    draw_panel(ax, g3, "Generation 3 · Q1+Q2 reward", "scored on Q1+Q2 (1–5), training judge",
               t, base_level=base3, xmax=10, xstep=2)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args(argv)
    panels = load()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        fig = build(theme, panels)
        out = args.out_dir / f"history_{theme}.png"
        fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
