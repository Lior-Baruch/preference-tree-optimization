"""Draw the two PTO method schematics into ``results/figures/``.

    pto_framework.png        one training iteration (redraw of the paper's Figure 1)
    pto_preference_tree.png  inside one branch point, greedy mode (redraw of the paper's Figure 2)

These are schematics, not results: nothing here reads data. The figures carry no in-figure title
(the README / caption provides it) but do carry their colour legend so they read standalone.
Parameters shown are those of the reported runs: M = 8, K = 5, MCL = 12, gpt-4o-mini.

Needs only matplotlib:  python scripts/make_schematics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
NAVY = "#1F3A5F"
GREEN = "#008A63"
RED = "#C04A1A"
GREY = "#5A5A5A"
EDGE = "#8A96A3"

# (fill, edge) per box role: green = data we produce, purple = an API model we call,
# blue = the policy being trained, orange = the oracle.
ROLE = {
    "source": ("#EAF1E7", "#5B8C5A"),
    "api": ("#EDE6F3", "#7B5EA7"),
    "policy": ("#DCEBF5", BLUE),
    "data": ("#DCEDE3", GREEN),
    "oracle": ("#FBE8D5", ORANGE),
    "neutral": ("#EFF2F6", EDGE),
    "chosen": ("#DCEDE3", GREEN),
    "reject": ("#F7E3DC", RED),
}

Box = tuple[float, float, float, float]


def node(ax, x, y, w, h, text, role="neutral", fs=8.5, bold=False, pad=0.28, lw=1.2) -> Box:
    """Rounded box centred on (x, y). Returns the box so arrows can anchor to its edges."""
    fill, edge = ROLE[role]
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={pad},rounding_size={min(w, h) * 0.18:.3f}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#22282F", zorder=3, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, a: Box, b: Box, side="auto", color=NAVY, lw=1.15, style="-|>", rad=0.0, ls="-"):
    """Arrow from box a to box b, anchored on facing edges."""
    (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
    if side == "auto":
        side = "h" if abs(bx0 - ax0) >= abs(by0 - ay0) else "v"
    if side == "h":
        sx, sy = ax0 + aw / 2 * (1 if bx0 > ax0 else -1), ay0
        ex, ey = bx0 - bw / 2 * (1 if bx0 > ax0 else -1), by0
    else:
        sx, sy = ax0, ay0 + ah / 2 * (1 if by0 > ay0 else -1)
        ex, ey = bx0, by0 - bh / 2 * (1 if by0 > ay0 else -1)
    ax.add_patch(FancyArrowPatch(
        (sx, sy), (ex, ey), arrowstyle=style, mutation_scale=9,
        color=color, linewidth=lw, linestyle=ls, zorder=1.5,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5))


def elbow(ax, pts, color=NAVY, lw=1.15, ls="-"):
    """Poly-line with an arrowhead on the last leg, for the feedback loop."""
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        ax.add_patch(FancyArrowPatch(
            pts[i], pts[i + 1], arrowstyle="-|>" if last else "-",
            mutation_scale=9 if last else 0, color=color, linewidth=lw, linestyle=ls,
            zorder=1.5, shrinkA=0, shrinkB=1.5 if last else 0))


def label(ax, x, y, text, fs=7.5, color=GREY, ha="center", va="center", bold=False,
          style="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, zorder=4,
            fontweight="bold" if bold else "normal", fontstyle=style, linespacing=1.35)


def bracket(ax, x, y0, y1, text, color=GREY, fs=7.5, tick=1.1):
    """Left-hand span bracket with a rotated label (the 'K look-ahead steps' device)."""
    ax.plot([x, x], [y0, y1], color=color, lw=1.0, zorder=1)
    ax.plot([x, x + tick], [y0, y0], color=color, lw=1.0, zorder=1)
    ax.plot([x, x + tick], [y1, y1], color=color, lw=1.0, zorder=1)
    ax.text(x - tick * 0.9, (y0 + y1) / 2, text, rotation=90, ha="center", va="center",
            fontsize=fs, color=color, zorder=4)


def canvas(w_in, h_in, xlim, ylim, legend=None, note=None):
    """A bare drawing surface with the colour legend (and an optional note) at the bottom."""
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.axis("off")
    y = ylim[0] + (ylim[1] - ylim[0]) * 0.035
    if note:
        ax.text(xlim[0] + 1, y, note, ha="left", va="bottom", fontsize=7.5,
                color=NAVY, fontstyle="italic")
        y += (ylim[1] - ylim[0]) * 0.055
    if legend:
        ax.text(xlim[0] + 1, y, legend, ha="left", va="bottom", fontsize=7.5, color=GREY)
    return fig, ax


def save(fig, out_dir: Path, name: str) -> None:
    out = out_dir / name
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def pto_framework(out_dir: Path) -> None:
    """One PTO iteration: policy + patient + oracle -> preference pairs -> DPO -> next policy."""
    fig, ax = canvas(11.4, 4.15, (0, 118), (-12, 47),
                     legend="Green = data this iteration produces · purple = an API model we call "
                            "· blue = the policy being trained · orange = the oracle",
                     note="The 96 conversations generated at the start of the iteration are also "
                          "the evaluation set for π$_n$ — there is no separate evaluation pass.")

    src = node(ax, 11, 26, 17, 11, "96 patient\nsystem prompts\n(one per persona)", "source", 8)
    orc = node(ax, 40, 41, 21, 8.5, "Oracle\ngpt-4o-mini · Q1+Q2", "oracle", 8)
    pat = node(ax, 40, 26, 21, 8.5, "Patient model\ngpt-4o-mini", "api", 8)
    pol = node(ax, 40, 11, 21, 8.5, "Policy  π$_n$\nLlama-3.2-1B + LoRA", "policy", 8)
    tree = node(ax, 72, 26, 20, 21,
                "Preference tree data\n\n(prompt, chosen, rejected)\n\nkept only where\n"
                "Δscore > τ", "data", 8)
    nxt = node(ax, 104, 26, 19, 8.5, "Policy  π$_{n+1}$", "policy", 8.5, bold=True)

    arrow(ax, src, pat)
    for b in (orc, pat, pol):
        arrow(ax, b, tree, side="h")
    arrow(ax, tree, nxt)
    label(ax, 88, 29.2, "DPO update", fs=8, color=NAVY, bold=True)

    # Feedback loop: the new policy becomes the next iteration's policy.
    elbow(ax, [(104, 21.4), (104, 3.0), (40, 3.0), (40, 6.4)])
    label(ax, 72, 4.6, "next iteration:  π$_{n+1}$ → π$_n$", fs=7.5)

    save(fig, out_dir, "pto_framework.png")


ROWS = dict(root=118, cand=105, la1=94, la2=84.5, dots=76.5, traj=67,
            oracle=55, score=44, agg=33, keep=21, reply=11.5, next=2.0)
COLS = (18, 50, 82)
TAGS = ("1", "2", "M")


def _generation_top(ax):
    """Rows above the oracle: trunk -> M candidates -> K-utterance look-ahead -> trajectories."""
    r = ROWS
    root = node(ax, 50, r["root"], 62, 8,
                "Trunk$_i$  —  conversation so far  (≥ MCL = 12 utterances)", "api", 8.5)
    bracket(ax, 3.5, r["la1"] + 4.2, r["dots"] - 3.0, "K = 5 look-ahead utterances", fs=7.5)
    label(ax, 8.5, r["cand"] + 7.0, "M = 8\ncandidates", fs=7.5, color=NAVY, bold=True)

    trajs = []
    for x, tag in zip(COLS, TAGS, strict=True):
        c = node(ax, x, r["cand"], 26, 7.2, f"candidate$_{{i,{tag}}}$", "data", 8)
        node(ax, x, r["la1"], 26, 6.2, "patient reply", "api", 7.5)
        node(ax, x, r["la2"], 26, 6.2, "therapist reply", "policy", 7.5)
        label(ax, x, r["dots"], "⋮", fs=11, color=GREY)
        t = node(ax, x, r["traj"], 26, 7.2, f"trajectory$_{{i,{tag}}}$", "api", 8)
        arrow(ax, root, c, side="v")
        trajs.append(t)
        for a, b in ((r["cand"], r["la1"]), (r["la1"], r["la2"])):
            ax.add_patch(FancyArrowPatch((x, a - 3.9), (x, b + 3.4), arrowstyle="-|>",
                                         mutation_scale=8, color=NAVY, lw=1.05, zorder=1.5))
        ax.add_patch(FancyArrowPatch((x, r["dots"] - 2.2), (x, r["traj"] + 3.9), arrowstyle="-|>",
                                     mutation_scale=8, color=NAVY, lw=1.05, zorder=1.5))

    orc = node(ax, 50, r["oracle"], 74, 7.6,
               "Oracle  —  gpt-4o-mini scores the FULL trajectory on Q1+Q2", "oracle", 8.5)
    for t in trajs:
        arrow(ax, t, orc, side="v")
    return orc


def pto_preference_tree(out_dir: Path) -> None:
    """Inside one greedy branch point: M candidates, look-ahead, scoring, tau-filtered pair."""
    r = ROWS
    fig, ax = canvas(9.6, 11.0, (-4, 120), (-24, 126),
                     legend="M candidates are generated, all M are scored, and exactly two of "
                            "them survive into the training set.")

    orc = _generation_top(ax)

    scores = []
    for x, tag in zip(COLS, TAGS, strict=True):
        sc = node(ax, x, r["score"], 24, 6.4, f"score$_{{i,{tag}}}$", "neutral", 8)
        arrow(ax, orc, sc, side="v")
        scores.append(sc)

    agg = node(ax, 50, r["agg"], 74, 7.4,
               "best and worst  ·  emit a pair ONLY if  score$_{best}$ − score$_{worst}$ > τ",
               "neutral", 8.5)
    for sc in scores:
        arrow(ax, sc, agg, side="v")

    rej = node(ax, 22, r["keep"], 32, 7.2, "rejected = worst", "reject", 8, bold=True)
    cho = node(ax, 70, r["keep"], 32, 7.2, "chosen = best", "chosen", 8, bold=True)
    arrow(ax, agg, rej, side="v")
    arrow(ax, agg, cho, side="v")

    # The DPO training pair is the only thing this branch point contributes to the dataset.
    pair = node(ax, 24, r["reply"] - 0.5, 46, 7.0,
                "training pair\n(trunk$_i$, chosen, rejected)  →  DPO", "data", 7.8, bold=True)
    arrow(ax, rej, pair, side="v")
    arrow(ax, cho, pair, side="h")

    rep = node(ax, 74, r["reply"], 28, 6.4, "patient reply", "api", 7.5)
    arrow(ax, cho, rep, side="v")
    nxt = node(ax, 60, r["next"], 62, 7.2,
               "Trunk$_{i+1}$  =  trunk$_i$  +  chosen  +  patient reply", "api", 8.5)
    arrow(ax, rep, nxt, side="v")

    for x in COLS:
        label(ax, x, -7.5, "⋮", fs=11, color=GREY)
        ax.add_patch(FancyArrowPatch((60, -1.7), (x, -5.6), arrowstyle="-|>", mutation_scale=8,
                                     color=NAVY, lw=1.05, zorder=1.5))
    label(ax, 50, -12.5, "branch again from the new trunk, until it reaches the target length",
          fs=7.5, color=GREY, style="italic")

    label(ax, 100, r["keep"] + 3.0,
          "greedy:\nthe winner is\nappended to the\ntrunk, so this\nchoice conditions\n"
          "the NEXT\nbranch point",
          fs=7.5, color=BLUE, ha="left", va="top", bold=True)
    save(fig, out_dir, "pto_preference_tree.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pto_framework(args.out_dir)
    pto_preference_tree(args.out_dir)


if __name__ == "__main__":
    main()
