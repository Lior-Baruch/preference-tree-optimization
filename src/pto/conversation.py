"""Conversation state, transcript formats and training-prompt construction.

Torch-free building blocks shared by conversation generation, preference-tree growth and the
look-ahead simulator:

- :class:`ConversationState` and its CSV round-trip (one utterance per row);
- ChatML control markers and :func:`clean_completion`;
- conversation init/update and ``SESSION ENDED`` handling (:func:`_process_session_response`);
- message-list views of role-tagged turns (therapist and patient perspective) and the plain-text
  transcript the oracle scores (:func:`format_conversation_for_oracle`);
- :func:`build_truncated_training_prompt`, which caps a therapist prompt to a token budget by
  dropping the oldest turns.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ChatML control markers. `<|im_end|>` ends a turn; `<|im_start|>` opens one.
# A base (non-instruct) model has never been trained on this template (these are NOT special
# tokens: they tokenize as ordinary BPE pieces), so early in training it frequently self-plays by
# emitting `<|im_start|>` and writing the *other* speaker's turn as literal text. Generation
# therefore (a) stops at the FIRST of either marker and (b) cuts any residual leak out of the
# decoded text. Keep these in sync with the chat template (``pto.model.CHATML_TEMPLATE``) and the
# configured stop strings.
_CHATML_MARKERS = ("<|im_end|>", "<|im_start|>")

# Default stop strings for therapist generation. `<|im_start|>` is included so a self-play
# attempt halts the moment the model opens a fake new turn. This prevents both degenerate turns
# made of repeated `<|im_start|>` and the role-swap derailment where a leaked first-person
# `<|im_start|>user` line flips the patient simulator into counselor mode for the rest of the
# conversation.
_DEFAULT_STOP_STRINGS = ["<|im_end|>", "<|im_start|>"]

SESSION_ENDED_KEYWORD = "SESSION ENDED"


def clean_completion(text: str | None) -> str:
    """Cut a raw therapist completion at the first ChatML control marker.

    Returns the content before the earliest `<|im_start|>`/`<|im_end|>` (stripped). With
    `<|im_start|>` in the stop strings this is usually a no-op, but it salvages any completion
    that overran (the generation stop and this cut use the same markers, so the saved
    conversation, the oracle transcript and every DPO pair are free of leaked role headers).
    An empty string signals a degenerate turn (the model produced only a marker / nothing usable).
    """
    if not text:
        return ""
    cut = len(text)
    for marker in _CHATML_MARKERS:
        i = text.find(marker)
        if i != -1 and i < cut:
            cut = i
    return text[:cut].strip()


# ---------------------------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------------------------


@dataclass
class ConversationState:
    """Mutable state of one conversation during batched synthesis.

    Two parallel representations are kept:

    - ``conversation``: flat list of utterance strings (therapist first, alternating);
    - ``turns``: explicit role-tagged turns ``{"role": "therapist"|"patient", "content": str}``.

    ``messages_*_assist`` are the chat histories in each model's role convention (the patient
    simulator sees the therapist as ``user``; the therapist policy sees the patient as ``user``).
    They are cleared once a conversation ends to save memory.

    ``permutation_index`` is the conversation's index within the (shuffled) persona list of the
    current iteration; it names the CSV file ``conversation_<permutation_index>.csv``.
    """

    permutation_index: int
    conversation: list  # flat list of utterance strings
    messages_Patient_assist: list  # [{"role": "system"|"user"|"assistant", "content": str}]
    messages_Therapist_assist: list  # [{"role": "system"|"user"|"assistant", "content": str}]
    turns: list = field(default_factory=list)  # [{"role": "therapist"|"patient", "content": str}]
    next_speaker: str = "patient"  # "therapist" or "patient": who speaks next
    session_ended_by: str | None = None
    session_ended_explanation: str | None = None
    is_active: bool = True
    failed: bool = False


# ---------------------------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------------------------


def _extract_unique_col(df: pd.DataFrame, col_name: str) -> str | None:
    """Extract a single unique non-null value from a DataFrame column, or None."""
    if col_name not in df.columns:
        return None
    vals = df[col_name].dropna().unique()
    return str(vals[0]) if len(vals) > 0 else None


def load_conversation_from_csv(
    csv_path: str | os.PathLike, permutation_index: int
) -> ConversationState:
    """Load a single conversation from a CSV file.

    If a ``role`` column exists, roles are read explicitly. Otherwise roles are inferred by
    alternating turns (therapist first). The returned state is inactive.
    """
    df = pd.read_csv(csv_path, keep_default_na=False)
    conversation = df["conversation"].tolist()

    if "role" in df.columns:
        turns = [
            {"role": str(r), "content": str(c)}
            for r, c in zip(df["role"], df["conversation"], strict=False)
        ]
    else:
        turns = [
            {"role": "therapist" if j % 2 == 0 else "patient", "content": str(c)}
            for j, c in enumerate(conversation)
        ]

    return ConversationState(
        permutation_index=permutation_index,
        conversation=conversation,
        messages_Patient_assist=[],
        messages_Therapist_assist=[],
        turns=turns,
        session_ended_by=_extract_unique_col(df, "session_ended_by"),
        session_ended_explanation=_extract_unique_col(df, "session_ended_explanation"),
        is_active=False,
    )


def save_conversation_csv(state: ConversationState, save_dir: str | os.PathLike) -> str:
    """Write one ConversationState to ``save_dir/conversation_<permutation_index>.csv``.

    Columns: ``role, conversation, session_ended_by, session_ended_explanation`` (one utterance
    per row). Roles come from ``state.turns`` when present, else are inferred by alternating
    (therapist first). Returns the path written.
    """
    save_path = os.path.join(save_dir, f"conversation_{state.permutation_index}.csv")
    roles = (
        [t["role"] for t in state.turns]
        if state.turns
        else ["therapist" if j % 2 == 0 else "patient" for j in range(len(state.conversation))]
    )
    pd.DataFrame({
        "role": roles,
        "conversation": state.conversation,
        "session_ended_by": state.session_ended_by,
        "session_ended_explanation": state.session_ended_explanation,
    }).to_csv(save_path, index=False)
    return save_path


# ---------------------------------------------------------------------------------------------
# Conversation init / update / session end
# ---------------------------------------------------------------------------------------------


def initialize_conversation(
    system_prompt_therapist: str,
    system_prompt_patient: str,
    therapist_init_utterance: str,
    patient_init_utterance: str,
    include_empty_init_user_message: bool = False,
) -> tuple[list, list, list]:
    """Initialize the utterance list and both assistants' message lists.

    Returns ``(conversation, messages_Patient_assist, messages_Therapist_assist)``.

    Args:
        include_empty_init_user_message: If True, include the initial user message (even if
            empty) before the therapist's first assistant message. Required by chat templates
            that expect system -> user -> assistant ordering. Set to False if the template
            supports system -> assistant directly.
    """
    conversation = [therapist_init_utterance]

    messages_Patient_assist = [
        {"role": "system", "content": system_prompt_patient},
        {"role": "user", "content": therapist_init_utterance},
    ]

    if include_empty_init_user_message or patient_init_utterance:
        messages_Therapist_assist = [
            {"role": "system", "content": system_prompt_therapist},
            {"role": "user", "content": patient_init_utterance},
            {"role": "assistant", "content": therapist_init_utterance},
        ]
    else:
        messages_Therapist_assist = [
            {"role": "system", "content": system_prompt_therapist},
            {"role": "assistant", "content": therapist_init_utterance},
        ]

    return conversation, messages_Patient_assist, messages_Therapist_assist


def update_conversation(
    conversation: list,
    messages_Patient_assist: list,
    messages_Therapist_assist: list,
    role_Patient: str,
    role_Therapist: str,
    response_content: str,
    turns: list | None = None,
    speaker_role: str | None = None,
) -> None:
    """Append a new response to the utterance list and both message lists.

    Args:
        turns: If provided, also append a role-tagged turn dict.
        speaker_role: "therapist" or "patient"; required when ``turns`` is provided.
    """
    conversation.append(response_content)
    messages_Patient_assist.append({"role": role_Patient, "content": response_content})
    messages_Therapist_assist.append({"role": role_Therapist, "content": response_content})
    if turns is not None and speaker_role is not None:
        turns.append({"role": speaker_role, "content": response_content})


def handle_session_end(response_content: str, speaker_role: str) -> tuple[str, str, str]:
    """Split a response at the ``SESSION ENDED`` keyword (case-insensitive).

    Returns ``(ended_by, ended_explanation, cleaned_response_content)``: the text after the
    keyword (e.g. the condition number) is the explanation, the text before it is kept as the
    speaker's last utterance. Raises ``ValueError`` if the keyword is absent.
    """
    idx = response_content.upper().find(SESSION_ENDED_KEYWORD)
    if idx == -1:
        raise ValueError("SESSION ENDED keyword not found in response content")

    ended_explanation = response_content[idx + len(SESSION_ENDED_KEYWORD):]
    cleaned = response_content[:idx]
    return speaker_role, ended_explanation, cleaned


def _process_session_response(
    state: ConversationState,
    response_content: str,
    speaker_role: str,
    role_Patient: str,
    role_Therapist: str,
) -> bool:
    """Apply one generated response to ``state`` and handle the terminal marker.

    If ``SESSION ENDED`` appears, the conversation is marked inactive and the trailing
    explanation is stored; any text before the marker is still kept as a valid turn. Returns
    ``True`` when generation should continue.

    A degenerate (empty after cleaning) turn ends the conversation instead of appending an empty
    utterance: :func:`clean_completion` cuts self-played ``<|im_start|>`` leaks, which can leave
    nothing usable. Ending here keeps the saved conversation clean instead of padding it with an
    empty turn or letting a leak derail the roles.
    """
    if not response_content or not response_content.strip():
        state.is_active = False
        return False

    if SESSION_ENDED_KEYWORD in response_content.upper():
        try:
            ended_by, ended_expl, response_content = handle_session_end(
                response_content, speaker_role
            )
            state.session_ended_by = ended_by
            state.session_ended_explanation = ended_expl
            state.is_active = False
            if response_content:
                update_conversation(
                    state.conversation, state.messages_Patient_assist,
                    state.messages_Therapist_assist, role_Patient,
                    role_Therapist, response_content,
                    turns=state.turns, speaker_role=speaker_role,
                )
        except ValueError as e:
            logger.warning(
                "SESSION ENDED handling failed for conversation %s: %s",
                state.permutation_index, e,
            )
            state.is_active = False
        return False

    update_conversation(
        state.conversation, state.messages_Patient_assist,
        state.messages_Therapist_assist, role_Patient,
        role_Therapist, response_content,
        turns=state.turns, speaker_role=speaker_role,
    )
    next_map = {"patient": "therapist", "therapist": "patient"}
    state.next_speaker = next_map[speaker_role]
    return True


# ---------------------------------------------------------------------------------------------
# Transcript / message views
# ---------------------------------------------------------------------------------------------


def format_conversation_for_oracle(messages: list[dict[str, Any]]) -> str:
    """Render therapist-perspective messages as the plain-text transcript the oracle scores.

    Uses ``[PATIENT]`` / ``[THERAPIST]`` labels (``user`` = patient, ``assistant`` = therapist),
    drops the system prompt and empty messages, and joins turns with a blank line.

    Note:
        Message content may include embedded newlines (including ``"\\n\\n"``).
        :func:`pto.lookahead._parse_transcript_to_messages` is the inverse parser and handles
        continuation fragments to reconstruct the original turn.
    """
    transcript = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            transcript.append(f"[PATIENT]: {content}")
        elif role == "assistant":
            transcript.append(f"[THERAPIST]: {content}")
        # 'system' messages are deliberately ignored.
    return "\n\n".join(transcript)


def turns_to_messages(turns: list[dict[str, Any]], system_prompt: str) -> list[dict[str, str]]:
    """Therapist-perspective view of role-tagged turns (therapist -> assistant, patient -> user).

    Args:
        turns: List of ``{"role": "therapist"|"patient", "content": str}`` dicts.
        system_prompt: System prompt for the therapist.

    Returns:
        Message dicts with role in {'system', 'assistant', 'user'}.
    """
    messages = [{"role": "system", "content": str(system_prompt)}]
    role_map = {"therapist": "assistant", "patient": "user"}
    for turn in turns:
        messages.append({"role": role_map[turn["role"]], "content": str(turn["content"])})
    return messages


def turns_to_patient_messages(
    turns: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, str]]:
    """Patient-perspective view of role-tagged turns (therapist -> user, patient -> assistant).

    Mirror of :func:`turns_to_messages` with the roles flipped and the patient's system prompt;
    matches the ``messages_Patient_assist`` shape built by :func:`initialize_conversation`.

    Args:
        turns: List of ``{"role": "therapist"|"patient", "content": str}`` dicts.
        system_prompt: System prompt for the patient.

    Returns:
        Message dicts with role in {'system', 'assistant', 'user'}.
    """
    messages = [{"role": "system", "content": str(system_prompt)}]
    role_map = {"therapist": "user", "patient": "assistant"}
    for turn in turns:
        messages.append({"role": role_map[turn["role"]], "content": str(turn["content"])})
    return messages


# ---------------------------------------------------------------------------------------------
# Token-budgeted training prompts
# ---------------------------------------------------------------------------------------------
#
# ``_estimate_turn_token_costs`` + ``_compute_system_overhead`` give cheap running totals so the
# drop point is found without re-rendering the template per candidate; the result is then
# verified with a single accurate render. The ChatML strings below must match
# ``pto.model.CHATML_TEMPLATE``.


def _estimate_turn_token_costs(turns: list[dict[str, Any]], tokenizer: Any) -> list[int]:
    """Estimate per-turn token costs by encoding each turn's ChatML wrapper.

    Estimates may differ slightly from a full template render due to token boundary effects at
    message joins, but are accurate enough for budget decisions.
    """
    role_map = {"therapist": "assistant", "patient": "user"}
    costs = []
    for turn in turns:
        role = role_map.get(turn["role"], turn["role"])
        single_msg = f"<|im_start|>{role}\n{turn['content']}<|im_end|>\n"
        costs.append(len(tokenizer.encode(single_msg, add_special_tokens=False)))
    return costs


def _compute_system_overhead(system_prompt: str, tokenizer: Any) -> int:
    """Token overhead of the system message plus the generation-prompt suffix."""
    sys_msg = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    overhead = len(tokenizer.encode(sys_msg, add_special_tokens=False))
    gen_suffix = "<|im_start|>assistant\n"
    overhead += len(tokenizer.encode(gen_suffix, add_special_tokens=False))
    return overhead


def _truncate_by_dropping_turns(
    partial_turns: list[dict[str, Any]],
    system_prompt: str,
    tokenizer: Any,
    max_prompt_tokens: int,
    turn_token_costs: list[int] | None = None,
    system_overhead: int | None = None,
) -> str | None:
    """Drop the oldest turns until the rendered prompt fits the token budget.

    Uses per-turn token cost estimates to find the drop point, then verifies with a single
    accurate render, falling back to iterative dropping if the estimate was slightly off.
    The most recent turn is always kept; returns ``None`` if even that does not fit.
    """
    if turn_token_costs is None:
        turn_token_costs = _estimate_turn_token_costs(partial_turns, tokenizer)
    if system_overhead is None:
        system_overhead = _compute_system_overhead(system_prompt, tokenizer)

    total = system_overhead + sum(turn_token_costs)

    drop = 0
    while total > max_prompt_tokens and drop < len(partial_turns) - 1:
        total -= turn_token_costs[drop]
        drop += 1

    if drop >= len(partial_turns):
        return None

    remaining = partial_turns[drop:]
    messages = turns_to_messages(remaining, system_prompt)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens:
        return prompt

    # The estimate was slightly off: fall back to dropping one turn at a time from here.
    for _ in range(1, len(remaining)):
        remaining = remaining[1:]
        if not remaining:
            return None
        messages = turns_to_messages(remaining, system_prompt)
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens:
            return prompt

    return None


def build_truncated_training_prompt(
    partial_turns: list[dict[str, Any]],
    system_prompt: str,
    tokenizer: Any,
    max_prompt_tokens: int,
    truncation_mode: str = "drop_oldest",
) -> str | None:
    """Render ``partial_turns`` as a chat-template therapist prompt capped to
    ``max_prompt_tokens`` by dropping the oldest turns (the system message and the most recent
    turns are always kept).

    This matters for two reasons: (1) it matches the inference-time therapist context window
    (``max_input_tokens``), so the policy is not trained on a context it never sees at
    generation time; (2) it keeps a long grown transcript from reaching ``DPOTrainer`` verbatim,
    which would exceed the DPO ``max_length`` and, with ``keep_start`` truncation, slice the
    *response* off the end (and inflate the LM-head logits memory).

    Returns ``None`` if even the single most recent turn exceeds the budget; the caller should
    skip the pair. Only ``truncation_mode="drop_oldest"`` is supported.
    """
    if truncation_mode != "drop_oldest":
        raise ValueError(
            f"Unsupported truncation_mode {truncation_mode!r}; only 'drop_oldest' is supported"
        )
    messages = turns_to_messages(partial_turns, system_prompt)
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens:
        return prompt
    return _truncate_by_dropping_turns(
        partial_turns, system_prompt, tokenizer, max_prompt_tokens
    )
