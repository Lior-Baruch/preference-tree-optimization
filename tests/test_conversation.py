"""Conversation state, CSV persistence, transcript views and token-budgeted DPO prompts."""

from __future__ import annotations

import pytest

from pto.conversation import (
    ConversationState,
    _process_session_response,
    build_truncated_training_prompt,
    clean_completion,
    format_conversation_for_oracle,
    handle_session_end,
    load_conversation_from_csv,
    save_conversation_csv,
    turns_to_messages,
    turns_to_patient_messages,
)

SYSTEM = "You are a motivational interviewing counselor named David."


def _turns(n: int, words: int = 3) -> list[dict[str, str]]:
    roles = ("therapist", "patient")
    return [
        {"role": roles[i % 2], "content": " ".join([f"utterance{i:02d}"] * words)}
        for i in range(n)
    ]


# --- clean_completion --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        ("", ""),
        ("   ", ""),
        ("How are you feeling today?", "How are you feeling today?"),
        ("  padded reply \n", "padded reply"),
        ("Tell me more.<|im_end|>\n<|im_start|>user\nI smoke.", "Tell me more."),
        ("What brings you here? <|im_start|>user\nI am James", "What brings you here?"),
        # The earliest marker wins, whichever it is.
        ("A<|im_start|>B<|im_end|>C", "A"),
        ("A<|im_end|>B<|im_start|>C", "A"),
        ("<|im_start|>user\nself-played patient turn", ""),
        ("<|im_end|>", ""),
    ],
)
def test_clean_completion(raw: str | None, expected: str) -> None:
    assert clean_completion(raw) == expected


# --- CSV round trip ----------------------------------------------------------------------------


def test_csv_round_trip(tmp_path) -> None:
    turns = [
        {"role": "therapist", "content": "Hello, what brings you here?"},
        {"role": "patient", "content": "NA"},  # must stay a string, not NaN
        {"role": "therapist", "content": 'Commas, "quotes"\nand newlines survive.'},
        {"role": "patient", "content": "None"},
    ]
    state = ConversationState(
        permutation_index=7,
        conversation=[t["content"] for t in turns],
        messages_Patient_assist=[],
        messages_Therapist_assist=[],
        turns=turns,
        session_ended_by="patient",
        session_ended_explanation=" 2. I am satisfied",
    )
    path = save_conversation_csv(state, str(tmp_path))
    assert path.endswith("conversation_7.csv")

    loaded = load_conversation_from_csv(path, 7)
    assert loaded.permutation_index == 7
    assert loaded.conversation == state.conversation
    assert loaded.turns == turns
    assert loaded.session_ended_by == "patient"
    assert loaded.session_ended_explanation == " 2. I am satisfied"
    assert loaded.is_active is False


def test_csv_without_role_column_alternates(tmp_path) -> None:
    import pandas as pd

    path = tmp_path / "conversation_0.csv"
    pd.DataFrame({"conversation": ["t0", "p1", "t2"]}).to_csv(path, index=False)
    loaded = load_conversation_from_csv(str(path), 0)
    assert [t["role"] for t in loaded.turns] == ["therapist", "patient", "therapist"]
    assert loaded.session_ended_by is None


# --- session handling --------------------------------------------------------------------------


def _fresh_state() -> ConversationState:
    return ConversationState(
        permutation_index=0,
        conversation=["Hello"],
        messages_Patient_assist=[
            {"role": "system", "content": "patient"},
            {"role": "user", "content": "Hello"},
        ],
        messages_Therapist_assist=[
            {"role": "system", "content": SYSTEM},
            {"role": "assistant", "content": "Hello"},
        ],
        turns=[{"role": "therapist", "content": "Hello"}],
        next_speaker="patient",
    )


def test_handle_session_end_splits_text() -> None:
    by, expl, cleaned = handle_session_end("Thanks, bye. session ended 2", "patient")
    assert (by, expl, cleaned) == ("patient", " 2", "Thanks, bye. ")
    with pytest.raises(ValueError):
        handle_session_end("no marker here", "patient")


def test_process_session_response_appends_and_flips_speaker() -> None:
    state = _fresh_state()
    assert _process_session_response(state, "I smoke a lot.", "patient", "assistant", "user")
    assert state.conversation[-1] == "I smoke a lot."
    assert state.turns[-1] == {"role": "patient", "content": "I smoke a lot."}
    assert state.messages_Patient_assist[-1] == {"role": "assistant", "content": "I smoke a lot."}
    assert state.messages_Therapist_assist[-1] == {"role": "user", "content": "I smoke a lot."}
    assert state.next_speaker == "therapist"
    assert state.is_active


def test_process_session_response_session_ended_keeps_prefix() -> None:
    state = _fresh_state()
    cont = _process_session_response(
        state, "I think I got what I needed. SESSION ENDED 2", "patient", "assistant", "user"
    )
    assert cont is False
    assert state.is_active is False
    assert state.session_ended_by == "patient"
    assert state.session_ended_explanation == " 2"
    assert state.conversation[-1] == "I think I got what I needed. "
    assert len(state.turns) == 2


def test_process_session_response_marker_only_appends_nothing() -> None:
    state = _fresh_state()
    assert not _process_session_response(state, "SESSION ENDED 1", "patient", "assistant", "user")
    assert state.is_active is False
    assert state.session_ended_by == "patient"
    assert state.conversation == ["Hello"]


@pytest.mark.parametrize("empty", ["", "   \n"])
def test_process_session_response_empty_ends_conversation(empty: str) -> None:
    state = _fresh_state()
    assert not _process_session_response(state, empty, "therapist", "user", "assistant")
    assert state.is_active is False
    assert state.session_ended_by is None
    assert state.conversation == ["Hello"]


# --- transcript / message views ----------------------------------------------------------------


def test_format_conversation_for_oracle() -> None:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "assistant", "content": " Hi, I am David. "},
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "   "},
        {"role": "user", "content": "Line one\n\nline two"},
    ]
    assert format_conversation_for_oracle(messages) == (
        "[THERAPIST]: Hi, I am David.\n\n[PATIENT]: Hi.\n\n[PATIENT]: Line one\n\nline two"
    )


def test_turn_views_are_mirrors() -> None:
    turns = _turns(3)
    therapist_view = turns_to_messages(turns, SYSTEM)
    patient_view = turns_to_patient_messages(turns, "patient prompt")
    assert [m["role"] for m in therapist_view] == ["system", "assistant", "user", "assistant"]
    assert [m["role"] for m in patient_view] == ["system", "user", "assistant", "user"]
    assert therapist_view[0]["content"] == SYSTEM
    assert patient_view[0]["content"] == "patient prompt"
    assert [m["content"] for m in therapist_view[1:]] == [t["content"] for t in turns]


def test_transcript_parser_inverts_formatter() -> None:
    from pto.lookahead import _count_role_labels, _parse_transcript_to_messages

    turns = [
        {"role": "therapist", "content": "Hello there."},
        {"role": "patient", "content": "First paragraph.\n\nSecond paragraph."},
        {"role": "therapist", "content": "I see."},
    ]
    messages = turns_to_messages(turns, SYSTEM)
    transcript = format_conversation_for_oracle(messages)
    assert _count_role_labels(transcript) == 3
    th, pt = _parse_transcript_to_messages(transcript, SYSTEM, "patient prompt")
    assert th == messages
    assert pt == turns_to_patient_messages(turns, "patient prompt")
    with pytest.raises(ValueError):
        _parse_transcript_to_messages("no label at start", SYSTEM, "p")


# --- build_truncated_training_prompt -----------------------------------------------------------


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def test_truncated_prompt_fits_without_truncation(tiny_tokenizer) -> None:
    turns = _turns(4)
    prompt = build_truncated_training_prompt(turns, SYSTEM, tiny_tokenizer, max_prompt_tokens=4096)
    expected = tiny_tokenizer.apply_chat_template(
        turns_to_messages(turns, SYSTEM), add_generation_prompt=True, tokenize=False
    )
    assert prompt == expected
    assert prompt.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n")
    assert prompt.endswith("<|im_end|>\n<|im_start|>assistant\n")


def test_truncated_prompt_drops_oldest_turns(tiny_tokenizer) -> None:
    turns = _turns(12, words=6)
    full = build_truncated_training_prompt(turns, SYSTEM, tiny_tokenizer, max_prompt_tokens=10**6)
    full_len = _n_tokens(tiny_tokenizer, full)
    budget = full_len // 2
    prompt = build_truncated_training_prompt(turns, SYSTEM, tiny_tokenizer, budget)
    assert prompt is not None
    assert _n_tokens(tiny_tokenizer, prompt) <= budget
    # System prompt and the most recent turn are always kept; the oldest turn is gone.
    assert prompt.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>\n")
    assert turns[-1]["content"] in prompt
    assert turns[0]["content"] not in prompt
    # What remains is a contiguous suffix of the conversation rendered by the template.
    kept = [t for t in turns if t["content"] in prompt]
    assert kept == turns[len(turns) - len(kept):]
    assert prompt == tiny_tokenizer.apply_chat_template(
        turns_to_messages(kept, SYSTEM), add_generation_prompt=True, tokenize=False
    )


def test_truncated_prompt_none_when_last_turn_too_long(tiny_tokenizer) -> None:
    turns = _turns(3)
    turns[-1] = {"role": "patient", "content": "word " * 400}
    system_only = tiny_tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}], add_generation_prompt=True, tokenize=False
    )
    budget = _n_tokens(tiny_tokenizer, system_only) + 5
    assert build_truncated_training_prompt(turns, SYSTEM, tiny_tokenizer, budget) is None
