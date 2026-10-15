"""K-turn look-ahead: extend each candidate therapist turn with simulated future turns.

With look-ahead ``K > 0`` a candidate is scored on ``prefix + candidate + K simulated turns``
(patient simulator via the API, therapist via the current policy) instead of on
``prefix + candidate`` alone, so the oracle rewards turns by their effect on where the
conversation goes rather than only on how they read in isolation.

All simulations advance in lock-step (patient, therapist, patient, ...), which turns ``B * K``
serial generations into about ``K`` batched ``model.generate`` calls.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from .conversation import handle_session_end
from .generation import generate_patient_responses_batch, generate_therapist_responses_batch

logger = logging.getLogger(__name__)

__all__ = ["LookaheadConfig", "simulate_lookahead_batch"]

# A transcript segment that starts with a role label: "[PATIENT]: ..." / "[THERAPIST]: ...".
_TRANSCRIPT_LINE_RE = re.compile(r"^\[(PATIENT|THERAPIST)\]:\s*(.*)$", re.DOTALL)


@dataclass(frozen=True)
class LookaheadConfig:
    """Look-ahead settings (``k`` extra simulated turns before oracle scoring)."""

    k: int  # 0 disables look-ahead
    temperature_therapist: float
    temperature_patient: float
    max_tokens: int
    max_input_tokens: int
    patient_model_id: str
    patient_api_concurrency: int
    patient_api_max_retries: int
    patient_api_backoff_seconds: float
    stop_strings: list[str] | None = None
    # Cap on the batched therapist generate inside the look-ahead. None = all active sims in
    # one padded generate call (fastest, most memory). Halved automatically on OOM.
    lookahead_sub_batch_size: int | None = None


def _count_role_labels(s: str) -> int:
    """Count ``[PATIENT]:`` / ``[THERAPIST]:`` labels in a transcript.

    Used to measure how many look-ahead turns a rollout actually realized beyond its prefix.
    """
    return s.count("[PATIENT]:") + s.count("[THERAPIST]:")


def _parse_transcript_to_messages(
    transcript: str,
    system_prompt_therapist: str,
    system_prompt_patient: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Parse a plain-text transcript back into the two chat-message views.

    Inverse of :func:`pto.conversation.format_conversation_for_oracle`. Turns are separated
    by ``\\n\\n`` and start with a ``[PATIENT]:`` / ``[THERAPIST]:`` label. Turn content may
    itself contain ``\\n\\n``, so an unlabeled fragment is a continuation of the previous
    labeled turn.

    Returns:
        ``(messages_therapist, messages_patient)``: therapist=assistant / patient=user in the
        first list, patient=assistant / therapist=user in the second.

    Raises:
        ValueError: If the transcript starts with an unlabeled segment.
    """
    messages_therapist = [{"role": "system", "content": system_prompt_therapist}]
    messages_patient = [{"role": "system", "content": system_prompt_patient}]

    segments = [s.strip() for s in transcript.split("\n\n") if s.strip()]

    current_label: str | None = None
    current_fragments: list[str] = []

    def _flush_current_turn() -> None:
        nonlocal current_label, current_fragments
        if current_label is None:
            return

        content = "\n\n".join(fragment for fragment in current_fragments if fragment).strip()
        if not content:
            current_label = None
            current_fragments = []
            return

        if current_label == "THERAPIST":
            messages_therapist.append({"role": "assistant", "content": content})
            messages_patient.append({"role": "user", "content": content})
        else:  # PATIENT
            messages_therapist.append({"role": "user", "content": content})
            messages_patient.append({"role": "assistant", "content": content})

        current_label = None
        current_fragments = []

    for segment in segments:
        m = _TRANSCRIPT_LINE_RE.match(segment)
        if m is not None:
            _flush_current_turn()
            current_label = m.group(1)
            current_fragments = [m.group(2).strip()]
            continue

        if current_label is None:
            raise ValueError(
                f"Cannot parse transcript segment without a preceding role label: {segment!r}"
            )

        current_fragments.append(segment)  # continuation of an embedded "\n\n"

    _flush_current_turn()
    return messages_therapist, messages_patient


def _therapist_generate_chunked(
    therapist_model: Any,
    therapist_tokenizer: Any,
    batch_messages: list[list[dict[str, str]]],
    max_tokens: int,
    temperature: float,
    max_input_tokens: int,
    stop_strings: list[str] | None,
    start_sub_batch: int,
) -> tuple[list[str | None], int, int]:
    """Generate one therapist turn per message list, halving the sub-batch on OOM.

    Synchronous; call it through ``run_in_executor`` while holding the GPU lock. It wraps
    :func:`pto.generation.generate_therapist_responses_batch`, which returns
    ``(responses | None, error_type)`` and never raises on OOM. Inputs are processed in
    chunks of ``sb`` (starting at ``start_sub_batch``):

    - success: responses are placed at their indices;
    - ``"oom"``: halve ``sb`` and retry the same chunk; at ``sb == 1`` that single item is
      frozen (``None``) and skipped. The reduced ``sb`` is sticky and returned, so later
      steps do not pay the OOM again;
    - ``"runtime_error"`` (non-OOM): halving cannot help, so the chunk's items are frozen and
      processing continues. Scoring a shorter transcript beats aborting the whole batch.

    Returns:
        ``(responses, final_sub_batch, n_generate_calls)``, responses aligned to the input.
    """
    n = len(batch_messages)
    responses: list[str | None] = [None] * n
    sb = max(1, start_sub_batch)
    n_calls = 0

    i = 0
    while i < n:
        chunk = batch_messages[i:i + sb]
        resp, error_type = generate_therapist_responses_batch(
            therapist_model, therapist_tokenizer, chunk,
            max_tokens, temperature,
            max_input_tokens=max_input_tokens, stop_strings=stop_strings,
        )
        n_calls += 1

        if error_type is None:
            for j, r in enumerate(resp):
                responses[i + j] = r
            i += len(chunk)
        elif error_type == "oom":
            if sb == 1:
                responses[i] = None  # a single sequence still OOMs: freeze it
                i += 1
            else:
                sb = max(1, sb // 2)  # retry the same start with the smaller (sticky) sb
        else:  # "runtime_error"
            for j in range(len(chunk)):
                responses[i + j] = None
            i += len(chunk)

    return responses, sb, n_calls


@dataclass
class _LookaheadSim:
    """Mutable per-candidate state of the lock-step look-ahead rollout."""

    msgs_therapist: list[dict[str, str]]
    msgs_patient: list[dict[str, str]]
    extended_transcript: str
    active: bool = True


async def simulate_lookahead_batch(
    transcripts: list[str],
    completions: list[str],
    system_prompt_therapist: str,
    system_prompts_patient: list[str],
    therapist_model: Any,
    therapist_tokenizer: Any,
    client: Any,
    patient_model_id: str,
    lookahead_k: int,
    temperature_therapist: float,
    temperature_patient: float,
    max_tokens: int,
    max_input_tokens: int,
    stop_strings: list[str] | None,
    patient_sem: asyncio.Semaphore,
    gpu_lock: asyncio.Lock,
    patient_max_retries: int = 3,
    patient_backoff_seconds: float = 1.0,
    sub_batch_size: int | None = None,
) -> list[str]:
    """Extend each ``(transcript, completion)`` pair by ``lookahead_k`` simulated turns.

    The completion is a therapist turn, so step 0 is a patient turn and the speaker
    alternates with the step index (even = patient, odd = therapist). Every simulation starts
    at the same phase and can only leave the cadence by becoming inactive, so speakers never
    desynchronize.

    Patient calls run concurrently (bounded by ``patient_sem``). Each therapist step is one
    batched generate over the active simulations, holding ``gpu_lock`` for that step only
    (never across a patient ``await``). The policy is switched to ``eval()`` with
    ``use_cache=True`` for the step and its previous ``train()`` / ``use_cache`` state is
    restored afterwards. OOM is handled by sub-batch halving.

    A simulation is frozen (kept at its current transcript) when its patient call fails,
    either side emits ``SESSION ENDED`` (the text before the keyword is kept), or the
    therapist turn is ``None`` / empty after cleaning. A transcript that fails to parse is
    frozen on ``transcript + completion`` instead of aborting the batch.

    Args:
        sub_batch_size: Cap on the therapist generate batch (None = all active simulations at
            once). Halved on OOM and kept for the remaining steps.

    Returns:
        Extended transcripts in input order, each starting with
        ``f"{transcript}\\n\\n[THERAPIST]: {completion}"``.
    """
    sims: list[_LookaheadSim] = []
    for transcript, completion, sp_patient in zip(
        transcripts, completions, system_prompts_patient
    ):
        seed_transcript = f"{transcript}\n\n[THERAPIST]: {completion}"
        try:
            msgs_therapist, msgs_patient = _parse_transcript_to_messages(
                transcript, system_prompt_therapist, sp_patient,
            )
            msgs_therapist.append({"role": "assistant", "content": completion})
            msgs_patient.append({"role": "user", "content": completion})
            sims.append(_LookaheadSim(msgs_therapist, msgs_patient, seed_transcript, True))
        except ValueError as e:
            preview = transcript[:280] + ("..." if len(transcript) > 280 else "")
            logger.warning(
                "Look-ahead transcript parse failed (length=%d chars); freezing this "
                "simulation on its seed transcript. Preview: %r (%s)",
                len(transcript), preview, e,
            )
            sims.append(_LookaheadSim([], [], seed_transcript, False))

    loop = asyncio.get_running_loop()
    sb = sub_batch_size  # halving inside the chunked helper persists across steps

    for step in range(lookahead_k):
        active = [s for s in sims if s.active]
        if not active:
            break

        if step % 2 == 0:  # patient turn
            responses = await generate_patient_responses_batch(
                client, patient_model_id,
                [s.msgs_patient for s in active],
                max_tokens, temperature_patient, patient_sem,
                patient_max_retries, patient_backoff_seconds,
            )
            for sim, resp in zip(active, responses):
                if isinstance(resp, BaseException) or resp is None:
                    sim.active = False
                    continue
                if "SESSION ENDED" in resp.upper():
                    try:
                        _, _, cleaned = handle_session_end(resp, "patient")
                        if cleaned:
                            sim.extended_transcript += f"\n\n[PATIENT]: {cleaned}"
                    except ValueError:
                        pass
                    sim.active = False
                    continue
                sim.msgs_therapist.append({"role": "user", "content": resp})
                sim.msgs_patient.append({"role": "assistant", "content": resp})
                sim.extended_transcript += f"\n\n[PATIENT]: {resp}"
        else:  # therapist turn: one batched generate under the GPU lock
            start_sb = sb if sb is not None else len(active)
            async with gpu_lock:
                old_use_cache = therapist_model.config.use_cache
                was_training = therapist_model.training
                therapist_model.config.use_cache = True
                therapist_model.eval()
                try:
                    therapist_responses, sb, _ = await loop.run_in_executor(
                        None, _therapist_generate_chunked,
                        therapist_model, therapist_tokenizer,
                        [s.msgs_therapist for s in active],
                        max_tokens, temperature_therapist, max_input_tokens,
                        stop_strings, start_sb,
                    )
                finally:
                    therapist_model.config.use_cache = old_use_cache
                    if was_training:
                        therapist_model.train()

            for sim, resp in zip(active, therapist_responses):
                if resp is None or not resp.strip():
                    # None = OOM / runtime failure; empty = degenerate turn (cleaning cut a
                    # self-played "<|im_start|>" leak down to nothing). Freeze either way.
                    sim.active = False
                    continue
                if "SESSION ENDED" in resp.upper():
                    try:
                        _, _, cleaned = handle_session_end(resp, "therapist")
                        if cleaned:
                            sim.extended_transcript += f"\n\n[THERAPIST]: {cleaned}"
                    except ValueError:
                        pass
                    sim.active = False
                    continue
                sim.msgs_therapist.append({"role": "assistant", "content": resp})
                sim.msgs_patient.append({"role": "user", "content": resp})
                sim.extended_transcript += f"\n\n[THERAPIST]: {resp}"

    return [s.extended_transcript for s in sims]
