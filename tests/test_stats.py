"""Paired statistics on synthetic score tables with a known effect."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pto.evaluation.stats import holm, paired_compare, summarize, to_markdown

N = 40


def _tidy(values: dict[str, np.ndarray], persona_ids: np.ndarray, offset: int = 0) -> pd.DataFrame:
    rows = []
    for inst, vals in values.items():
        for k, (pid, v) in enumerate(zip(persona_ids, vals)):
            rows.append(
                {"conversation_index": k + offset, "persona_id": int(pid), "instrument": inst,
                 "value": float(v)}
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(123)
    personas = np.arange(N)
    base_q1 = rng.normal(3.0, 0.5, N)
    # Dyadic values keep +-0.25 exact, so every |difference| ties and the null is exact.
    base_q2 = rng.integers(8, 40, N) / 8.0
    effect = 0.5 + rng.normal(0.0, 0.2, N)
    null = np.where(np.arange(N) % 2 == 0, 0.25, -0.25)  # exactly zero mean change
    a = _tidy({"Q1": base_q1, "Q2": base_q2}, personas)
    b_vals = {"Q1": base_q1 + effect, "Q2": base_q2 + null}
    # b lists personas in another order with other conversation indices: pairing is by persona.
    perm = rng.permutation(N)
    b = _tidy({k: v[perm] for k, v in b_vals.items()}, personas[perm], offset=1000)
    # A persona scored only in ``a`` must not enter the comparison.
    extra = pd.DataFrame(
        [{"conversation_index": 999, "persona_id": 500, "instrument": "Q1", "value": 1.0}]
    )
    a = pd.concat([a, extra], ignore_index=True)
    return a, b, effect, null


def test_paired_compare_known_effect(synthetic) -> None:
    a, b, effect, null = synthetic
    out = paired_compare(a, b, n_boot=2000, seed=0)
    assert list(out.columns) == [
        "instrument", "n", "mean_a", "mean_b", "delta", "d_z", "ci_low", "ci_high",
        "p_wilcoxon", "p_holm",
    ]
    res = out.set_index("instrument")

    q1 = res.loc["Q1"]
    assert q1["n"] == N
    assert q1["delta"] == pytest.approx(effect.mean(), abs=1e-9)
    assert q1["delta"] == pytest.approx(q1["mean_b"] - q1["mean_a"], abs=1e-9)
    assert q1["d_z"] == pytest.approx(effect.mean() / effect.std(ddof=1), rel=1e-6)
    assert q1["ci_low"] < q1["delta"] < q1["ci_high"]
    assert q1["ci_low"] > 0.3
    assert q1["p_wilcoxon"] < 1e-6

    q2 = res.loc["Q2"]
    assert q2["n"] == N
    assert q2["delta"] == pytest.approx(0.0, abs=1e-9)
    assert q2["ci_low"] < 0 < q2["ci_high"]
    assert q2["p_wilcoxon"] > 0.5

    # Holm across the instruments of one comparison.
    p = res["p_wilcoxon"].to_numpy()
    order = np.argsort(p)
    expected = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        expected[i] = running
    np.testing.assert_allclose(res["p_holm"].to_numpy(), expected)
    assert (res["p_holm"] >= res["p_wilcoxon"]).all()


def test_paired_compare_is_reproducible(synthetic) -> None:
    a, b, _, _ = synthetic
    first = paired_compare(a, b, n_boot=500, seed=7)
    second = paired_compare(a, b, n_boot=500, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_summarize(synthetic) -> None:
    a, _, _, _ = synthetic
    out = summarize(a).set_index("instrument")
    q2 = a.loc[a["instrument"] == "Q2", "value"].to_numpy()
    assert out.loc["Q2", "n"] == N
    assert out.loc["Q2", "mean"] == pytest.approx(q2.mean())
    assert out.loc["Q2", "se"] == pytest.approx(q2.std(ddof=1) / np.sqrt(N))
    assert out.loc["Q1", "n"] == N + 1


def test_summarize_drops_undefined_values() -> None:
    df = pd.DataFrame(
        {"conversation_index": [0, 1, 2], "persona_id": [0, 1, 2], "instrument": ["PCT"] * 3,
         "value": [0.5, np.nan, 1.0]}
    )
    out = summarize(df).set_index("instrument")
    assert out.loc["PCT", "n"] == 2
    assert out.loc["PCT", "mean"] == pytest.approx(0.75)


def test_to_markdown(synthetic) -> None:
    a, b, _, _ = synthetic
    md = to_markdown(paired_compare(a, b, n_boot=200))
    lines = md.strip().splitlines()
    assert lines[0].startswith("| instrument")
    assert len(lines) == 2 + 2  # header, separator, Q1, Q2
    assert any(line.startswith("| Q1 ") for line in lines)
    assert "<0.001" in md


def test_holm_keeps_nan_out_of_the_family() -> None:
    np.testing.assert_allclose(holm([0.01, np.nan, 0.04]), [0.02, np.nan, 0.04])


def test_wilcoxon_convention_drops_zero_differences() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    # Zero differences are where the conventions differ: "wilcox" drops them, "pratt" and
    # "zsplit" keep them in the ranking.
    diff = np.array([0, 0, 0, 0.5, 0.3, -0.1, 0.7, 0.2, 0.4, -0.2, 0.6, 0.1])
    personas = np.arange(diff.size)
    a_vals = np.full(diff.size, 3.0)
    a = _tidy({"Q1": a_vals}, personas)
    b = _tidy({"Q1": a_vals + diff}, personas, offset=100)
    p = paired_compare(a, b, n_boot=200)["p_wilcoxon"].iloc[0]
    expected = scipy_stats.wilcoxon(diff, zero_method="wilcox").pvalue
    assert p == pytest.approx(expected)
    assert p != pytest.approx(scipy_stats.wilcoxon(diff, zero_method="pratt").pvalue)
