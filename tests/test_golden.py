"""Measurement instruments must be byte-identical to the originals they were ported from.

``tests/fixtures/golden_prompts.json`` was produced by ``tests/fixtures/make_golden.py`` from the
original modules; any change to a persona prompt, the therapist prompt, a questionnaire text or
the oracle prompt layout breaks comparability with the reported scores and fails here.
"""

from __future__ import annotations

import json

import pytest

from pto import personas, questionnaires


def test_persona_count_and_order(golden: dict) -> None:
    grid = personas.get_personas()
    assert len(grid) == golden["num_personas"] == 96
    prompts = [p["patient_system_prompt"] for p in grid]
    assert prompts == golden["patient_system_prompts"]


def test_persona_args_match(golden: dict) -> None:
    grid = personas.get_personas()
    assert [p["args"] for p in grid] == golden["persona_args"]
    # Key order is part of the serialized metadata, keep it too.
    assert json.dumps([p["args"] for p in grid]) == json.dumps(golden["persona_args"])


def test_persona_prompts_are_unique(golden: dict) -> None:
    assert len(set(golden["patient_system_prompts"])) == 96


def test_counselor_prompt_in_grid(golden: dict) -> None:
    grid = personas.get_personas()
    assert sorted({p["counselor_system_prompt"] for p in grid}) == golden[
        "counselor_system_prompts"
    ]


def test_therapist_prompts(golden: dict) -> None:
    system_prompt, init_utterance = personas.get_therapist_prompts()
    assert system_prompt == golden["therapist_system_prompt"]
    assert init_utterance == golden["therapist_init_utterance"]


def test_setup_permutations_matches_golden(golden: dict) -> None:
    from pto.model import setup_permutations

    perms, system_prompt, init_utterance = setup_permutations()
    assert [p["patient_system_prompt"] for p in perms] == golden["patient_system_prompts"]
    assert system_prompt == golden["therapist_system_prompt"]
    assert init_utterance == golden["therapist_init_utterance"]


@pytest.mark.parametrize("qid", ["1", "2", "3", "4", "6", "7", "8", "9", "10"])
def test_oracle_prompt_and_schema(golden: dict, qid: str) -> None:
    expected = golden["questionnaires"][qid]
    ed = questionnaires.get_prompt_eval_questionnaire(
        questionnaire=int(qid), conversation=golden["sample_transcript"]
    )
    assert ed["prompt"] == expected["prompt"]
    assert ed["questions_count"] == expected["questions_count"]
    assert list(ed["labels"]) == expected["labels"]
    assert ed["scale_min"] == expected["scale_min"]
    assert ed["scale_max"] == expected["scale_max"]
    assert ed["schema"] == expected["schema"]
    # Same key order as well: the schema is sent verbatim in every request.
    assert json.dumps(ed["schema"]) == json.dumps(expected["schema"])


def test_oracle_prompt_layout_transcript_last(golden: dict) -> None:
    """Instructions + rubric first, transcript last (shared prefix for prompt caching)."""
    transcript = golden["sample_transcript"]
    for qid in (1, 2):
        prompt = questionnaires.get_prompt_eval_questionnaire(qid, transcript)["prompt"]
        assert prompt.endswith(f"{transcript}\n\n--------------------")
        assert prompt.index("Transcript:") > prompt.index("IMPORTANT")
