"""Judge-based scoring of a conversation directory (fake judge clients, no network)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pto.evaluation import INSTRUMENTS, JudgeSpec, evaluate_run
from pto.questionnaires import _count_therapist_utterances, get_prompt_eval_questionnaire

PT_CYCLE = ("CT", "ST", "CT", "NEU")
NO_TALK_MARKER = "nothing to report today"


def _conversations() -> list[list[str]]:
    convs = []
    for c in range(3):
        utts = []
        for j in range(7 + c):  # odd/even lengths -> different therapist counts
            speaker = "Therapist" if j % 2 == 0 else "Patient"
            utts.append(f"{speaker} line {j} of conversation {c}.")
        convs.append(utts)
    return convs


def _transcript(utterances: list[str]) -> str:
    return "\n".join(
        f"[{'THERAPIST' if i % 2 == 0 else 'PATIENT'}]: {u}" for i, u in enumerate(utterances)
    )


def _controlled_judge(schema_tools):
    """Likert items get ``lo + qid % span``; behaviour counts are 1; MIPROC codes are scripted."""

    def answer(schema: dict, prompt: str) -> dict:
        qid = schema_tools.qid(schema)
        if qid == 10:
            props = schema["properties"]
            n_th = props["therapist_codes"]["minItems"]
            n_pt = props["patient_codes"]["minItems"]
            if NO_TALK_MARKER in prompt:
                pt = ["NEU"] * n_pt
            else:
                pt = [PT_CYCLE[i % len(PT_CYCLE)] for i in range(n_pt)]
            return {"questionnaire_id": 10, "therapist_codes": ["OQ"] * n_th, "patient_codes": pt}

        def pick(lo: int, hi: int | None, path: str) -> int:
            if hi is None:  # a behaviour count
                return 1
            return lo + qid % (hi - lo + 1)

        return schema_tools.instance(schema, prompt, int_fn=pick)

    return answer


def _expected_likert(qid: int, transcript: str) -> float:
    ed = get_prompt_eval_questionnaire(qid, transcript)
    lo, hi = ed["scale_min"], ed["scale_max"]
    return float(lo + qid % (hi - lo + 1))


def _expected_pct(n_patient: int) -> float:
    codes = [PT_CYCLE[i % len(PT_CYCLE)] for i in range(n_patient)]
    ct, st = codes.count("CT"), codes.count("ST")
    return ct / (ct + st)


@pytest.fixture
def conv_dir(tmp_path, conversation_dir_writer):
    return conversation_dir_writer(tmp_path / "model_iter_0", _conversations(), [40, 3, 77])


def test_all_instruments_with_controlled_judge(conv_dir, fake_openai, schema_tools, no_sleep):
    client = fake_openai(oracle_fn=_controlled_judge(schema_tools))
    judge = JudgeSpec.parse("openai:gpt-4o-mini-2024-07-18")
    df = evaluate_run(conv_dir, judge, client=client, concurrency=4)

    assert list(df.columns) == ["conversation_index", "persona_id", "instrument", "value"]
    assert set(df["instrument"]) == set(INSTRUMENTS) | {"Q1+Q2"}
    assert len(df) == 3 * (len(INSTRUMENTS) + 1)
    assert dict(zip(df["conversation_index"], df["persona_id"])) == {0: 40, 1: 3, 2: 77}

    # PCT comes from the MIPROC coder (questionnaire 10); the PCT rubric (8) is never called.
    qids = client.oracle_qids()
    assert 10 in qids and 8 not in qids
    assert sorted(set(qids)) == [1, 2, 3, 4, 6, 7, 9, 10]

    convs = _conversations()
    for i, utts in enumerate(convs):
        transcript = _transcript(utts)
        got = df[df["conversation_index"] == i].set_index("instrument")["value"]
        q1, q2 = _expected_likert(1, transcript), _expected_likert(2, transcript)
        assert got["Q1"] == pytest.approx(q1)
        assert got["Q2"] == pytest.approx(q2)
        assert got["Q1+Q2"] == pytest.approx((q1 + q2) / 2)
        assert got["WAI-SR"] == pytest.approx(_expected_likert(3, transcript))
        assert got["CSQ-8"] == pytest.approx(_expected_likert(4, transcript))
        assert got["MI-SAT"] == pytest.approx(_expected_likert(6, transcript))
        assert got["MITI"] == pytest.approx(_expected_likert(7, transcript))
        # MICI rate = inconsistent behaviours (6 categories x 1) per therapist utterance.
        n_th = _count_therapist_utterances(transcript)
        assert n_th == (len(utts) + 1) // 2
        assert got["MICI"] == pytest.approx(6 / n_th)
        assert got["PCT"] == pytest.approx(_expected_pct(len(utts) // 2))


def test_openai_judge_request(conv_dir, fake_openai, schema_tools, no_sleep):
    client = fake_openai(oracle_fn=_controlled_judge(schema_tools))
    judge = JudgeSpec.parse("openai:gpt-4o-mini-2024-07-18")
    evaluate_run(conv_dir, judge, instruments=["Q1"], client=client)

    convs = _conversations()
    prompts = set()
    for call in client.calls:
        assert call["model"] == "gpt-4o-mini-2024-07-18"
        assert call["temperature"] == pytest.approx(0.1)
        assert call["seed"] == 42
        assert call["response_format"]["json_schema"]["strict"] is True
        assert call["response_format"]["json_schema"]["name"] == "questionnaire_1_evaluation"
        (msg,) = call["messages"]
        assert msg["role"] == "user"
        prompts.add(msg["content"])
    # Judge transcripts: one utterance per line, therapist first (as the reported scores).
    assert prompts == {get_prompt_eval_questionnaire(1, _transcript(u))["prompt"] for u in convs}


def test_pct_undefined_without_change_or_sustain_talk(
    tmp_path, conversation_dir_writer, fake_openai, schema_tools, no_sleep
):
    convs = [["Hello.", f"Well, {NO_TALK_MARKER}.", "I see.", "Okay."]]
    conv_dir = conversation_dir_writer(tmp_path / "convs", convs, [5])
    client = fake_openai(oracle_fn=_controlled_judge(schema_tools))
    df = evaluate_run(conv_dir, JudgeSpec.parse("openai:m"), instruments=["PCT"], client=client)
    pct = df.loc[df["instrument"] == "PCT", "value"]
    # An undefined PCT is a NaN row, unlike a failed judge call, which leaves no row.
    assert len(pct) == 1 and math.isnan(pct.iloc[0])


def test_resume_skips_scored_pairs(conv_dir, tmp_path, fake_openai, schema_tools, no_sleep):
    out_csv = tmp_path / "scores.csv"
    judge = JudgeSpec.parse("openai:gpt-4o-mini-2024-07-18")
    instruments = ["Q1", "Q2", "MICI", "PCT"]
    first = fake_openai(oracle_fn=_controlled_judge(schema_tools))
    df1 = evaluate_run(conv_dir, judge, instruments=instruments, out_csv=out_csv, client=first)
    assert len(first.calls) == 3 * len(instruments)
    assert out_csv.exists()

    second = fake_openai(oracle_fn=_controlled_judge(schema_tools))
    df2 = evaluate_run(conv_dir, judge, instruments=instruments, out_csv=out_csv, client=second)
    assert second.calls == []
    key = ["conversation_index", "instrument"]
    pd.testing.assert_frame_equal(
        df1.sort_values(key).reset_index(drop=True), df2.sort_values(key).reset_index(drop=True)
    )

    # Drop the Q2 rows: only those calls are repeated.
    stored = pd.read_csv(out_csv)
    stored[stored["instrument"] != "Q2"].to_csv(out_csv, index=False)
    third = fake_openai(oracle_fn=_controlled_judge(schema_tools))
    df3 = evaluate_run(conv_dir, judge, instruments=instruments, out_csv=out_csv, client=third)
    assert third.oracle_qids() == [2, 2, 2]
    assert (df3["instrument"] == "Q2").sum() == 3


def test_failed_calls_are_absent_not_zero(conv_dir, fake_openai, no_sleep):
    client = fake_openai(oracle_fn=lambda schema, prompt: "not json")
    df = evaluate_run(conv_dir, JudgeSpec.parse("openai:m"), instruments=["Q1"], client=client)
    assert df.empty
    assert len(client.calls) == 3 * 3  # three conversations x three attempts


def test_anthropic_judge_strips_constraints(conv_dir, fake_anthropic, no_sleep):
    client = fake_anthropic()
    judge = JudgeSpec.parse("anthropic:claude-haiku-4-5")
    df = evaluate_run(conv_dir, judge, instruments=["Q1", "PCT"], client=client)
    assert set(df["instrument"]) == {"Q1", "PCT"}
    assert len(df) == 6
    for call in client.calls:
        assert call["model"] == "claude-haiku-4-5"
        assert "temperature" not in call
        schema = call["output_config"]["format"]["schema"]
        text = repr(schema)
        for key in ("minimum", "maximum", "minItems", "maxItems"):
            assert f"'{key}'" not in text
        assert "EXACTLY" in text


def test_judge_spec_parse() -> None:
    spec = JudgeSpec.parse("anthropic:claude-haiku-4-5")
    assert (spec.provider, spec.model) == ("anthropic", "claude-haiku-4-5")
    spec = JudgeSpec.parse("openai:gpt-4o-mini-2024-07-18")
    assert (spec.provider, spec.model, spec.temperature, spec.seed) == (
        "openai", "gpt-4o-mini-2024-07-18", 0.1, 42,
    )
    with pytest.raises(ValueError):
        JudgeSpec.parse("bogus:model")


def test_unknown_instrument_rejected(conv_dir, fake_openai) -> None:
    with pytest.raises(ValueError):
        evaluate_run(conv_dir, JudgeSpec.parse("openai:m"), instruments=["NOPE"],
                     client=fake_openai())
