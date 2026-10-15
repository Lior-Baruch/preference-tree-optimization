"""Summaries and persona-paired comparisons of tidy score tables.

Inputs are the tables produced by :func:`pto.evaluation.score.evaluate_run`
(``conversation_index, persona_id, instrument, value``). Two runs are compared by pairing their
conversations on ``persona_id``: every model state is evaluated on the same persona grid, so the
per-persona difference removes persona-level variance.

Conventions: ``delta = mean(b - a)``; Cohen's ``d_z = mean(diff) / std(diff, ddof=1)``;
percentile-bootstrap 95% CI of the mean difference; two-sided Wilcoxon signed-rank test
(``zero_method="wilcox"``); Holm-Bonferroni adjustment across the instruments of one comparison.
Statistics need at least 3 pairs; below that they are NaN.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .score import COLUMNS, COMPOSITE, INSTRUMENTS

_ORDER = [COMPOSITE, *INSTRUMENTS]
_MIN_PAIRS = 3


def _check(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    missing = [c for c in ("persona_id", "instrument", "value") if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing column(s) {missing}; expected {COLUMNS}.")
    out = df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])


def _ordered(instruments: Sequence[str]) -> list[str]:
    known = [i for i in _ORDER if i in instruments]
    return known + sorted(i for i in instruments if i not in _ORDER)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument ``n``, ``mean`` and standard error (``std(ddof=1) / sqrt(n)``).

    Undefined values (NaN, e.g. PCT with no change or sustain talk) are excluded.
    """
    df = _check(df)
    rows = []
    for inst in _ordered(df["instrument"].unique().tolist()):
        v = df.loc[df["instrument"] == inst, "value"].to_numpy(float)
        n = int(v.size)
        se = float(v.std(ddof=1) / np.sqrt(n)) if n >= 2 else np.nan
        rows.append({"instrument": inst, "n": n, "mean": float(v.mean()) if n else np.nan,
                     "se": se})
    return pd.DataFrame(rows, columns=["instrument", "n", "mean", "se"])


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaNs preserved and not counted)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    m = idx.size
    prev = 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (m - rank) * p[i]))
        out[i] = prev
    return out


def _bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = diff.size
    means = np.empty(n_boot)
    # Chunked so large tables do not allocate an n_boot x n index matrix at once.
    chunk = max(1, min(n_boot, 2_000_000 // max(n, 1)))
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _wilcoxon_p(diff: np.ndarray) -> float:
    try:
        from scipy import stats
    except ImportError as e:
        raise ImportError(
            "pto compare needs scipy for the Wilcoxon test. Install the 'eval' extra, "
            "e.g. pip install -e \".[eval]\" (or pip install scipy)."
        ) from e

    if not np.any(diff != 0):
        return 1.0
    try:
        return float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


def _per_persona(df: pd.DataFrame, inst: str) -> pd.Series:
    sub = df[df["instrument"] == inst]
    return sub.groupby("persona_id")["value"].mean()


def paired_compare(
    a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 10000, seed: int = 0
) -> pd.DataFrame:
    """Compare two score tables instrument by instrument, paired by ``persona_id``.

    If a persona has several conversations in one table, their mean is used. Only personas
    scored in both tables enter a comparison (``n``).

    Returns:
        DataFrame with columns ``instrument, n, mean_a, mean_b, delta, d_z, ci_low, ci_high,
        p_wilcoxon, p_holm``; ``delta`` is ``b - a``.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    a, b = _check(a, "a"), _check(b, "b")
    shared = set(a["instrument"]) & set(b["instrument"])
    rows = []
    for inst in _ordered(list(shared)):
        pa, pb = _per_persona(a, inst), _per_persona(b, inst)
        common = pa.index.intersection(pb.index)
        va = pa.loc[common].to_numpy(float)
        vb = pb.loc[common].to_numpy(float)
        diff = vb - va
        n = int(diff.size)
        row = {
            "instrument": inst,
            "n": n,
            "mean_a": float(va.mean()) if n else np.nan,
            "mean_b": float(vb.mean()) if n else np.nan,
            "delta": float(diff.mean()) if n else np.nan,
            "d_z": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_wilcoxon": np.nan,
        }
        if n >= _MIN_PAIRS:
            sd = diff.std(ddof=1)
            row["d_z"] = float(diff.mean() / sd) if sd > 0 else np.nan
            row["ci_low"], row["ci_high"] = _bootstrap_ci(diff, n_boot, seed)
            row["p_wilcoxon"] = _wilcoxon_p(diff)
        rows.append(row)
    cols = ["instrument", "n", "mean_a", "mean_b", "delta", "d_z", "ci_low", "ci_high",
            "p_wilcoxon", "p_holm"]
    out = pd.DataFrame(rows, columns=cols[:-1])
    out["p_holm"] = holm(out["p_wilcoxon"].to_numpy(float)) if len(out) else []
    return out[cols]


def _fmt(col: str, value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if col.startswith("p_"):
            return "<0.001" if value < 0.001 else f"{value:.3f}"
        return f"{value:.3f}"
    return str(value)


def to_markdown(df: pd.DataFrame) -> str:
    """Render a summary or comparison table as a GitHub-flavoured Markdown table.

    Floats get three decimals; p-value columns (``p_*``) below 0.001 print as ``<0.001``.
    """
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" if c == "instrument" else "---:" for c in cols) + "|",
    ]
    for rec in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(c, v) for c, v in zip(cols, rec, strict=True)) + " |")
    return "\n".join(lines) + "\n"
