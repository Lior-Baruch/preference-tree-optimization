"""Conversation generation with the tiny policy and the fake patient (CPU, no network)."""

from __future__ import annotations

import asyncio

import pandas as pd

from pto.generation import (
    generate_all_conversations,
    generate_therapist_responses_batch,
    run_async,
)
from pto.personas import get_personas, get_therapist_prompts

PATIENT_MODEL = "gpt-4o-mini-2024-07-18"


def _generate(model, tokenizer, client, save_dir, *, n_convs=3, num_utterances=6, **kwargs):
    therapist_prompt, init_utterance = get_therapist_prompts()
    return generate_all_conversations(
        therapist_model=model,
        therapist_tokenizer=tokenizer,
        client=client,
        permutations=get_personas()[:n_convs],
        therapist_system_prompt=therapist_prompt,
        therapist_init_utterance=init_utterance,
        save_dir=save_dir,
        patient_model_id=PATIENT_MODEL,
        max_tokens_per_response=8,
        num_utterances=num_utterances,
        batch_size=2,
        patient_api_concurrency=4,
        patient_api_max_retries=1,
        patient_api_backoff_seconds=0.0,
        batch_cooldown_seconds=0.0,
        **kwargs,
    )


def test_therapist_batch_is_clean(make_tiny_model, tiny_tokenizer) -> None:
    therapist_prompt, init_utterance = get_therapist_prompts()
    messages = [
        {"role": "system", "content": therapist_prompt},
        {"role": "assistant", "content": init_utterance},
        {"role": "user", "content": "I smoke a pack a day."},
    ]
    responses, error = generate_therapist_responses_batch(
        make_tiny_model(), tiny_tokenizer, [messages, messages],
        max_tokens=8, temperature=0.9, max_input_tokens=2048,
        stop_strings=["<|im_end|>", "<|im_start|>"],
    )
    assert error is None
    assert len(responses) == 2
    for r in responses:
        assert isinstance(r, str)
        assert "<|im_start|>" not in r and "<|im_end|>" not in r
        assert r == r.strip()


def test_generate_save_and_resume(tmp_path, make_tiny_model, tiny_tokenizer, fake_openai) -> None:
    model = make_tiny_model()
    client = fake_openai()
    save_dir = tmp_path / "model_iter_0"
    states = _generate(model, tiny_tokenizer, client, str(save_dir), patient_api_seed=43)
    _, init_utterance = get_therapist_prompts()
    personas = get_personas()

    assert sorted(s.permutation_index for s in states) == [0, 1, 2]
    for s in states:
        # The scripted opening line plus up to ``num_utterances`` generated utterances.
        assert 2 <= len(s.conversation) <= 7
        assert s.conversation[0] == init_utterance
        roles = [t["role"] for t in s.turns]
        assert roles == ["therapist" if j % 2 == 0 else "patient" for j in range(len(roles))]
        csv = pd.read_csv(save_dir / f"conversation_{s.permutation_index}.csv",
                          keep_default_na=False)
        assert csv["conversation"].tolist() == s.conversation
        assert csv["role"].tolist() == roles

    # Every patient request carries its persona prompt, the patient model and the seed.
    assert client.patient_calls and not client.oracle_calls
    persona_prompts = {p["patient_system_prompt"] for p in personas[:3]}
    for call in client.patient_calls:
        assert call["model"] == PATIENT_MODEL
        assert call["seed"] == 43
        assert call["temperature"] == 0.7
        assert call["messages"][0]["role"] == "system"
        assert call["messages"][0]["content"] in persona_prompts
        assert call["messages"][1] == {"role": "user", "content": init_utterance}

    # A second call loads every conversation from disk and makes no API calls.
    again = fake_openai()
    resumed = _generate(model, tiny_tokenizer, again, str(save_dir), patient_api_seed=43)
    assert again.calls == []
    by_idx = {s.permutation_index: s for s in states}
    for s in resumed:
        assert s.conversation == by_idx[s.permutation_index].conversation
        assert s.turns == by_idx[s.permutation_index].turns


def test_patient_session_end(tmp_path, make_tiny_model, tiny_tokenizer, fake_openai) -> None:
    client = fake_openai(end_after_patient_turns=2)
    states = _generate(
        make_tiny_model(), tiny_tokenizer, client, str(tmp_path), n_convs=2, num_utterances=12
    )
    assert any(len(s.conversation) == 4 for s in states)
    for s in states:
        assert len(s.conversation) <= 4
        if len(s.conversation) == 4:  # the therapist did not end it first with an empty turn
            assert s.session_ended_by == "patient"
            assert s.session_ended_explanation == " 2"
            assert "SESSION ENDED" not in s.conversation[-1]
            csv = pd.read_csv(tmp_path / f"conversation_{s.permutation_index}.csv")
            assert set(csv["session_ended_by"]) == {"patient"}


def test_run_async_inside_running_loop() -> None:
    async def inner() -> int:
        await asyncio.sleep(0)
        return 41

    async def outer() -> int:
        # Notebook situation: a loop is already running in this thread.
        return run_async(inner()) + 1

    assert run_async(inner()) == 41
    assert asyncio.run(outer()) == 42
