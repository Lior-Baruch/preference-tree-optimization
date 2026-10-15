"""Oracle call: request layout, JSON validation, retry and back-off semantics."""

from __future__ import annotations

import asyncio
import json

import pytest

from pto.oracle import OracleAsyncPrimitives, get_evaluation_json
from pto.questionnaires import get_prompt_eval_questionnaire

TRANSCRIPT = "[THERAPIST]: Hello, what brings you here?\n\n[PATIENT]: I want to stop smoking."
COMPLETION = "What makes this the right time for you?"
FULL = f"{TRANSCRIPT}\n\n[THERAPIST]: {COMPLETION}"


def _evaluate(client, oracle_cfg, qid: int = 1, **kwargs):
    primitives = OracleAsyncPrimitives(oracle_concurrency=4, lookahead_patient_concurrency=4)
    return asyncio.run(
        get_evaluation_json(
            client, oracle_cfg, primitives,
            transcript=TRANSCRIPT, completion=COMPLETION, questionnaire_id=qid, **kwargs,
        )
    )


@pytest.mark.parametrize("qid", [1, 2])
def test_valid_response_and_request_layout(fake_openai, make_oracle_cfg, no_sleep, qid) -> None:
    client = fake_openai()
    cfg = make_oracle_cfg(model_id="oracle-model", eval_temperature=0.0)
    data, n_questions, attempts = _evaluate(client, cfg, qid=qid)

    expected = get_prompt_eval_questionnaire(questionnaire=qid, conversation=FULL)
    assert n_questions == expected["questions_count"]
    assert attempts == 1
    assert data["questionnaire_id"] == qid
    assert len(data["scores"]) == n_questions
    assert data["mean_score"] == pytest.approx(sum(data["scores"]) / n_questions)
    assert no_sleep == []

    (call,) = client.calls
    assert call["model"] == "oracle-model"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 256
    # Single user message: instructions + rubric first, transcript last (prompt caching).
    assert call["messages"] == [{"role": "user", "content": expected["prompt"]}]
    rf = call["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == f"questionnaire_{qid}_evaluation"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == expected["schema"]


def test_full_conversation_override(fake_openai, make_oracle_cfg, no_sleep) -> None:
    client = fake_openai()
    override = FULL + "\n\n[PATIENT]: Maybe my kids."
    data, _, _ = _evaluate(client, make_oracle_cfg(), full_conversation_override=override)
    assert data is not None
    expected = get_prompt_eval_questionnaire(questionnaire=1, conversation=override)["prompt"]
    assert client.calls[0]["messages"][0]["content"] == expected


def test_scores_depend_on_transcript(fake_openai, make_oracle_cfg, no_sleep) -> None:
    """Sanity check of the fake: different candidates must be able to clear the tau filter."""
    client = fake_openai()
    means = {
        _evaluate(client, make_oracle_cfg(), qid=2,
                  full_conversation_override=f"{FULL} variant {i}")[0]["mean_score"]
        for i in range(6)
    }
    assert len(means) > 1


@pytest.mark.parametrize(
    "bad",
    [
        {"questionnaire_id": 2, "scores": [3, 3, 3, 3, 3]},  # wrong questionnaire id
        {"questionnaire_id": 1, "scores": [3, 3, 3, 3, 6]},  # above scale
        {"questionnaire_id": 1, "scores": [0, 3, 3, 3, 3]},  # below scale
        {"questionnaire_id": 1, "scores": [3, 3, 3, 3]},  # wrong item count
        {"questionnaire_id": 1, "scores": [3, 3, 3, 3, 3.0]},  # non-integer score
        "",  # empty content
        "not json",
    ],
)
def test_invalid_responses_exhaust_retries(fake_openai, make_oracle_cfg, no_sleep, bad) -> None:
    client = fake_openai(oracle_fn=lambda schema, prompt: bad)
    cfg = make_oracle_cfg(max_retries=3)
    data, n_questions, attempts = _evaluate(client, cfg, qid=1)
    assert data is None
    assert n_questions == 5
    assert attempts == 3
    assert len(client.calls) == 3
    assert no_sleep == [1, 2]  # exponential back-off 2**attempt between attempts


def test_retry_then_success(fake_openai, make_oracle_cfg, no_sleep) -> None:
    client = fake_openai(fail_first=1)
    data, _, attempts = _evaluate(client, make_oracle_cfg(max_retries=3))
    assert data is not None
    assert attempts == 2
    assert no_sleep == [1]


def test_invalid_then_valid(fake_openai, make_oracle_cfg, no_sleep) -> None:
    answers = iter(["{oops", json.dumps({"questionnaire_id": 1, "scores": [5, 4, 4, 4, 3]})])
    client = fake_openai(oracle_fn=lambda schema, prompt: next(answers))
    data, _, attempts = _evaluate(client, make_oracle_cfg(max_retries=3))
    assert attempts == 2
    assert data["mean_score"] == pytest.approx(4.0)


def test_type_error_is_not_retried(fake_openai, make_oracle_cfg, no_sleep) -> None:
    client = fake_openai(oracle_fn=lambda schema, prompt: {"questionnaire_id": 1, "scores": 5})
    data, n_questions, attempts = _evaluate(client, make_oracle_cfg(max_retries=3))
    assert (data, n_questions, attempts) == (None, 5, 1)
    assert len(client.calls) == 1
    assert no_sleep == []


def test_primitives_are_per_loop() -> None:
    primitives = OracleAsyncPrimitives(oracle_concurrency=2, lookahead_patient_concurrency=3)

    async def grab():
        return (
            primitives.oracle_sem(),
            primitives.oracle_sem(),
            primitives.gpu_lock(),
            primitives.lookahead_patient_sem(),
        )

    a_sem, a_sem_again, a_lock, a_pat = asyncio.run(grab())
    b_sem, _, b_lock, _ = asyncio.run(grab())
    assert a_sem is a_sem_again
    assert a_sem is not b_sem and a_lock is not b_lock
    assert isinstance(a_lock, asyncio.Lock)
    assert isinstance(a_pat, asyncio.Semaphore)
