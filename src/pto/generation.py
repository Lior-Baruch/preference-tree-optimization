"""Conversation generation: policy therapist (local, batched) vs. simulated patient (API).

- :func:`generate_patient_response_async` / :func:`generate_patient_responses_batch`: patient
  turns via an OpenAI-compatible async chat-completions client, with retry/backoff.
- :func:`generate_therapist_responses_batch`: one batched ``model.generate`` call for the
  therapist turns, stopped at ChatML markers and cleaned with :func:`clean_completion`.
- :func:`conversation_loop_batch`: lock-step loop over a batch of conversations, with an
  OOM-safe therapist step (:func:`_therapist_step_oom_safe`).
- :func:`generate_all_conversations`: the full driver, with per-CSV resume from disk, retry
  passes and batch-size backoff on CUDA OOM.
- :func:`run_async` (re-exported from :mod:`pto._async`): runs a coroutine from synchronous
  code, also inside Jupyter.

The client object is always passed in explicitly; this module never builds one.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from . import _compat  # noqa: F401  (imports trl before torch; must stay above torch)

# isort: split

import torch

from ._async import run_async
from .conversation import (
    _DEFAULT_STOP_STRINGS,
    ConversationState,
    _process_session_response,
    clean_completion,
    initialize_conversation,
    load_conversation_from_csv,
    save_conversation_csv,
)

logger = logging.getLogger(__name__)


def _free_cuda_memory() -> None:
    """``gc.collect()`` and, when CUDA is available, release unused cached CUDA blocks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom_message(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


# ---------------------------------------------------------------------------------------------
# Patient (async API)
# ---------------------------------------------------------------------------------------------


async def generate_patient_response_async(
    client: Any,
    model_id: str,
    messages_Patient_assist: list,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    seed: int | None = None,
) -> str:
    """Generate one patient response via the async chat-completions API with retry/backoff.

    Retries any exception up to ``max_retries`` attempts in total, sleeping
    ``backoff_seconds * 2 ** (attempt - 1)`` between attempts; raises ``RuntimeError`` after the
    last failure.
    """
    last_error: BaseException | None = None
    api_kwargs: dict[str, Any] = dict(
        model=model_id,
        messages=messages_Patient_assist,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if seed is not None:
        api_kwargs["seed"] = seed
    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                response = await client.chat.completions.create(**api_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            sleep_s = backoff_seconds * (2 ** (attempt - 1))
            logger.warning(
                "Patient API attempt %d/%d failed: %s. Retrying in %.1fs...",
                attempt, max_retries, e, sleep_s,
            )
            await asyncio.sleep(sleep_s)

    raise RuntimeError(f"Patient API failed after {max_retries} retries: {last_error}")


async def generate_patient_responses_batch(
    client: Any,
    model_id: str,
    batch_messages: list[list],
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    seed: int | None = None,
) -> list:
    """Generate patient responses for a batch of conversations concurrently.

    Returns one entry per input, in order: the response string, or the exception that ended
    that request (``asyncio.gather(..., return_exceptions=True)``).
    """
    tasks = [
        generate_patient_response_async(
            client, model_id, messages, max_tokens, temperature,
            semaphore, max_retries, backoff_seconds, seed=seed,
        )
        for messages in batch_messages
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------------------------
# Therapist (batched local generation)
# ---------------------------------------------------------------------------------------------


def generate_therapist_responses_batch(
    therapist_model: Any,
    therapist_tokenizer: Any,
    batch_messages: list[list],
    max_tokens: int,
    temperature: float,
    max_input_tokens: int = 4096,
    stop_strings: list[str] | None = None,
) -> tuple[list[str] | None, str | None]:
    """Generate one therapist response per conversation with a single batched ``generate``.

    Prompts are rendered with the tokenizer's chat template (no BOS: ``add_special_tokens=False``)
    and left-truncated to ``max_input_tokens``. Generation stops at the first stop string; the
    ``tokenizer=`` argument is required by ``generate`` for string stopping criteria. Decoded
    text is passed through :func:`clean_completion`.

    Returns:
        ``(responses, error_type)``: ``responses`` is a list of strings on success, else None;
        ``error_type`` is one of None, ``"oom"``, ``"runtime_error"``.
    """
    if stop_strings is None:
        stop_strings = _DEFAULT_STOP_STRINGS

    prompts = [
        therapist_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in batch_messages
    ]

    encoded = None
    outputs = None
    try:
        encoded = therapist_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,
        ).to(therapist_model.device)

        with torch.inference_mode():
            outputs = therapist_model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                do_sample=True,
                max_new_tokens=max_tokens,
                pad_token_id=therapist_tokenizer.eos_token_id,
                eos_token_id=therapist_tokenizer.eos_token_id,
                temperature=temperature,
                num_return_sequences=1,
                stop_strings=stop_strings,
                tokenizer=therapist_tokenizer,
            )
    except torch.OutOfMemoryError as e:
        logger.warning("CUDA OOM during therapist generation: %s", e)
        _free_cuda_memory()
        return None, "oom"
    except RuntimeError as e:
        if _is_cuda_oom_message(e):
            logger.warning("Runtime CUDA memory failure during therapist generation: %s", e)
            _free_cuda_memory()
            return None, "oom"
        logger.error("Runtime error during therapist generation: %s", e)
        return None, "runtime_error"

    padded_input_length = encoded.input_ids.shape[1]

    responses = []
    for i in range(len(batch_messages)):
        new_tokens = outputs[i][padded_input_length:]
        decoded = therapist_tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(clean_completion(decoded))

    del encoded, outputs

    return responses, None


def _therapist_step_oom_safe(
    therapist_model: Any,
    therapist_tokenizer: Any,
    batch_messages: list[list],
    max_tokens: int,
    temperature: float,
    max_input_tokens: int,
    stop_strings: list[str] | None,
    sub_batch: int | None,
) -> tuple[list[str | None] | None, int | None, str | None]:
    """One therapist step for every active conversation, surviving CUDA OOM.

    With ``sub_batch`` None (no OOM yet in this batch) this is exactly one
    :func:`generate_therapist_responses_batch` call over all inputs. On ``"oom"`` it retries the
    SAME step in chunks, halving the chunk size until it fits, and returns that size so the
    caller reuses it (sticky) for the rest of the batch. No in-flight conversation is discarded,
    so long sessions are not under-sampled after an OOM. Mirrors
    ``pto.lookahead._therapist_generate_chunked``.

    Returns ``(responses, sub_batch, error_type)``: ``responses`` is order-aligned, with ``None``
    for an input that OOMs even alone; ``responses`` is ``None`` only on a non-OOM failure (the
    caller then aborts the batch).
    """
    n = len(batch_messages)
    if sub_batch is None or sub_batch >= n:
        resp, error_type = generate_therapist_responses_batch(
            therapist_model, therapist_tokenizer, batch_messages, max_tokens, temperature,
            max_input_tokens=max_input_tokens, stop_strings=stop_strings,
        )
        if error_type is None:
            return resp, sub_batch, None
        if error_type != "oom":
            return None, sub_batch, error_type
        if n == 1:
            return [None], 1, "oom"
        sub_batch = max(1, n // 2)
        logger.warning("CUDA OOM at therapist batch %d: retrying this step in chunks of %d",
                       n, sub_batch)

    responses: list[str | None] = [None] * n
    i = 0
    while i < n:
        chunk = batch_messages[i:i + sub_batch]
        resp, error_type = generate_therapist_responses_batch(
            therapist_model, therapist_tokenizer, chunk, max_tokens, temperature,
            max_input_tokens=max_input_tokens, stop_strings=stop_strings,
        )
        if error_type is None:
            responses[i:i + len(chunk)] = resp
            i += len(chunk)
        elif error_type == "oom":
            if sub_batch == 1:
                i += 1  # this one fails alone (stays None)
            else:
                sub_batch = max(1, sub_batch // 2)
                logger.warning("CUDA OOM: therapist chunk size halved to %d", sub_batch)
        else:
            return None, sub_batch, error_type
    return responses, sub_batch, None


# ---------------------------------------------------------------------------------------------
# Batched conversation loop
# ---------------------------------------------------------------------------------------------


async def conversation_loop_batch(
    batch_states: list[ConversationState],
    therapist_model: Any,
    therapist_tokenizer: Any,
    client: Any,
    semaphore: asyncio.Semaphore,
    num_utterances: int = 49,
    patient_model_id: str = "gpt-4o-mini-2024-07-18",
    max_tokens_per_response: int = 200,
    temperature_patient: float = 0.7,
    temperature_therapist: float = 0.9,
    patient_api_max_retries: int = 3,
    patient_api_backoff_seconds: float = 1.0,
    patient_api_seed: int | None = None,
    therapist_max_input_tokens: int = 2048,
    stop_strings: list[str] | None = None,
    verbose_detailed: bool = False,
) -> tuple[list[ConversationState], str | None, list[int]]:
    """Run a lock-step batched conversation loop over the active states.

    Each step processes only active conversations and expects the same ``next_speaker`` across
    them. If they desynchronize, only the desynced conversations are ended (their valid partial
    data is kept). ``num_utterances`` bounds the number of loop steps (one utterance each).

    Returns:
        ``(batch_states, error_type, desynced_indices)``: the states (updated in place); None on
        success, else an error category (``"oom"`` when a conversation OOMed even alone); and
        the permutation indices ended by desync.
    """
    desynced_indices: list[int] = []
    therapist_sub_batch: int | None = None  # set (sticky) once a therapist step hits OOM
    oom_failed = False
    for turn_num in range(num_utterances):
        active_states = [s for s in batch_states if s.is_active]
        if not active_states:
            break

        # All active states advance in lock step, so they share the next speaker.
        speaker_role = active_states[0].next_speaker
        desynced = [s for s in active_states if s.next_speaker != speaker_role]
        if desynced:
            ids = [(s.permutation_index, s.next_speaker) for s in desynced]
            logger.warning(
                "Speaker desync detected at turn %d. Expected '%s', desynced: %s. Ending desynced "
                "conversations gracefully (data up to this point is kept).",
                turn_num, speaker_role, ids,
            )
            for s in desynced:
                s.is_active = False
                # Not failed: the turns collected so far are valid.
                desynced_indices.append(s.permutation_index)
            active_states = [s for s in active_states if s.is_active]
            if not active_states:
                break

        if speaker_role == "patient":
            role_Patient = "assistant"
            role_Therapist = "user"

            batch_patient_messages = [s.messages_Patient_assist for s in active_states]
            responses = await generate_patient_responses_batch(
                client, patient_model_id, batch_patient_messages,
                max_tokens_per_response, temperature_patient,
                semaphore, patient_api_max_retries, patient_api_backoff_seconds,
                seed=patient_api_seed,
            )

            for state, response_content in zip(active_states, responses, strict=False):
                # Per-conversation API failure: that conversation fails alone.
                if isinstance(response_content, BaseException):
                    logger.warning("Patient API failed for conversation %s: %s",
                                   state.permutation_index, response_content)
                    state.is_active = False
                    state.failed = True
                    continue

                _process_session_response(
                    state, response_content, speaker_role, role_Patient, role_Therapist
                )
        else:  # speaker_role == "therapist"
            role_Patient = "user"
            role_Therapist = "assistant"

            batch_therapist_messages = [s.messages_Therapist_assist for s in active_states]
            responses, therapist_sub_batch, error_type = _therapist_step_oom_safe(
                therapist_model, therapist_tokenizer, batch_therapist_messages,
                max_tokens_per_response, temperature_therapist,
                max_input_tokens=therapist_max_input_tokens,
                stop_strings=stop_strings,
                sub_batch=therapist_sub_batch,
            )

            if responses is None:  # a non-OOM failure: abort the batch
                for state in active_states:
                    state.is_active = False
                    state.failed = True
                return (batch_states, (error_type or "therapist_generation_failed"),
                        desynced_indices)

            for state, response_content in zip(active_states, responses, strict=False):
                if response_content is None:  # OOM even on its own: this conversation fails alone
                    state.is_active = False
                    state.failed = True
                    oom_failed = True
                    continue
                _process_session_response(
                    state, response_content, speaker_role, role_Patient, role_Therapist
                )

        if verbose_detailed:
            truncated = [
                r[:100] + "..." if isinstance(r, str) and len(r) > 100 else r for r in responses
            ]
            logger.debug("%s RESPONSES (%d): %s", speaker_role.upper(), len(responses), truncated)

        # Release the message histories of conversations that just ended (CPU memory only).
        newly_ended = [s for s in active_states if not s.is_active]
        for s in newly_ended:
            s.messages_Patient_assist = []
            s.messages_Therapist_assist = []

    # A conversation that OOMed even at therapist sub-batch 1 failed alone; report "oom" so the
    # caller's pass-level backoff sees it (the other conversations of the batch are complete).
    return batch_states, ("oom" if oom_failed else None), desynced_indices


async def synthesize_conversations_batch(
    permutations_batch: list[tuple[int, dict[str, Any]]],
    system_prompt_therapist: str,
    therapist_init_utterance: str,
    therapist_model: Any,
    therapist_tokenizer: Any,
    client: Any,
    patient_api_concurrency: int = 16,
    include_empty_init_user_message: bool = False,
    num_utterances: int = 49,
    patient_model_id: str = "gpt-4o-mini-2024-07-18",
    max_tokens_per_response: int = 200,
    temperature_patient: float = 0.7,
    temperature_therapist: float = 0.9,
    patient_api_max_retries: int = 3,
    patient_api_backoff_seconds: float = 1.0,
    patient_api_seed: int | None = None,
    therapist_max_input_tokens: int = 2048,
    stop_strings: list[str] | None = None,
    verbose_detailed: bool = False,
) -> tuple[list[ConversationState], str | None, list[int]]:
    """Synthesize a batch of conversations in parallel.

    ``permutations_batch`` is a list of ``(index, persona)`` pairs; each persona provides its
    ``patient_system_prompt``. Every conversation starts with the therapist's scripted opening
    line, and the patient speaks next.
    """
    semaphore = asyncio.Semaphore(patient_api_concurrency)
    batch_states: list[ConversationState] = []
    for perm_idx, perm in permutations_batch:
        # patient_init_utterance is always empty: the patient speaks only after seeing the
        # therapist's greeting. With include_empty_init_user_message=True an empty user message
        # is inserted before the therapist's first assistant message, for chat templates that
        # require system -> user -> assistant ordering.
        conversation, messages_Patient, messages_Therapist = initialize_conversation(
            system_prompt_therapist,
            perm["patient_system_prompt"],
            therapist_init_utterance,
            "",
            include_empty_init_user_message=include_empty_init_user_message,
        )
        batch_states.append(
            ConversationState(
                permutation_index=perm_idx,
                conversation=conversation,
                messages_Patient_assist=messages_Patient,
                messages_Therapist_assist=messages_Therapist,
                turns=[{"role": "therapist", "content": therapist_init_utterance}],
                next_speaker="patient",
            )
        )

    batch_states, error_type, desynced_indices = await conversation_loop_batch(
        batch_states, therapist_model, therapist_tokenizer, client, semaphore,
        num_utterances=num_utterances,
        patient_model_id=patient_model_id,
        max_tokens_per_response=max_tokens_per_response,
        temperature_patient=temperature_patient,
        temperature_therapist=temperature_therapist,
        patient_api_max_retries=patient_api_max_retries,
        patient_api_backoff_seconds=patient_api_backoff_seconds,
        patient_api_seed=patient_api_seed,
        therapist_max_input_tokens=therapist_max_input_tokens,
        stop_strings=stop_strings,
        verbose_detailed=verbose_detailed,
    )
    return batch_states, error_type, desynced_indices


def run_synthesis_batch(
    permutations_batch: list[tuple[int, dict[str, Any]]],
    therapist_model: Any,
    therapist_tokenizer: Any,
    client: Any,
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    patient_api_concurrency: int = 16,
    include_empty_init_user_message: bool = False,
    num_utterances: int = 49,
    patient_model_id: str = "gpt-4o-mini-2024-07-18",
    max_tokens_per_response: int = 200,
    temperature_patient: float = 0.7,
    temperature_therapist: float = 0.9,
    patient_api_max_retries: int = 3,
    patient_api_backoff_seconds: float = 1.0,
    patient_api_seed: int | None = None,
    therapist_max_input_tokens: int = 2048,
    stop_strings: list[str] | None = None,
    verbose_detailed: bool = False,
) -> dict[str, Any]:
    """Execute one batch synthesis (on a fresh event loop) with normalized error handling.

    Returns:
        dict with keys ``final_states`` (list of ConversationState, or None on failure),
        ``error_type`` (None, or one of ``"oom"``, ``"runtime_error"``, ``"batch_exception"``,
        ``"therapist_generation_failed"``) and ``desynced_indices`` (permutation indices ended
        by desync).
    """
    try:
        final_states, error_type, desynced_indices = run_async(
            synthesize_conversations_batch(
                permutations_batch=permutations_batch,
                system_prompt_therapist=therapist_system_prompt,
                therapist_init_utterance=therapist_init_utterance,
                therapist_model=therapist_model,
                therapist_tokenizer=therapist_tokenizer,
                client=client,
                patient_api_concurrency=patient_api_concurrency,
                include_empty_init_user_message=include_empty_init_user_message,
                num_utterances=num_utterances,
                patient_model_id=patient_model_id,
                max_tokens_per_response=max_tokens_per_response,
                temperature_patient=temperature_patient,
                temperature_therapist=temperature_therapist,
                patient_api_max_retries=patient_api_max_retries,
                patient_api_backoff_seconds=patient_api_backoff_seconds,
                patient_api_seed=patient_api_seed,
                therapist_max_input_tokens=therapist_max_input_tokens,
                stop_strings=stop_strings,
                verbose_detailed=verbose_detailed,
            )
        )
        return {
            "final_states": final_states,
            "error_type": error_type,
            "desynced_indices": desynced_indices,
        }
    except torch.OutOfMemoryError as e:
        logger.error("CUDA OOM in batch generation: %s", e)
        _free_cuda_memory()
        return {"final_states": None, "error_type": "oom", "desynced_indices": []}
    except RuntimeError as e:
        if _is_cuda_oom_message(e):
            logger.error("Runtime CUDA memory failure in batch generation: %s", e)
            _free_cuda_memory()
            return {"final_states": None, "error_type": "oom", "desynced_indices": []}
        logger.error("Batch generation runtime error: %s", e)
        return {"final_states": None, "error_type": "runtime_error", "desynced_indices": []}
    except Exception as e:
        logger.error("Batch generation failed: %s", e)
        return {"final_states": None, "error_type": "batch_exception", "desynced_indices": []}


# ---------------------------------------------------------------------------------------------
# Full generation driver
# ---------------------------------------------------------------------------------------------


@dataclass
class _Counters:
    session_ended: int = 0
    completed: int = 0
    failed: int = 0
    desynced: list[int] = field(default_factory=list)


def _resume_from_disk(
    save_dir: str | os.PathLike, total: int, verbose: bool = False
) -> dict[int, ConversationState]:
    """Load previously saved ``conversation_<i>.csv`` files for ``i < total``.

    A CSV is written only once its conversation has finished, so an existing file means that
    index is done. Returns ``{index: state}``.
    """
    states: dict[int, ConversationState] = {}
    for i in range(total):
        csv_path = os.path.join(save_dir, f"conversation_{i}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            states[i] = load_conversation_from_csv(csv_path, i)
        except Exception as e:
            logger.warning("Could not load conversation_%d.csv: %s", i, e)
    if states and verbose:
        logger.info("Resumed %d/%d conversations from %s", len(states), total, save_dir)
    return states


def _record_completed_state(
    state: ConversationState,
    save_dir: str | os.PathLike | None,
    counters: _Counters,
    detailed: bool,
) -> bool:
    """Update counters and (optionally) save the CSV for one finished state.

    Returns True if the state should be kept; False if skipped as failed/empty (a conversation
    with no utterance beyond the therapist's opening line). In both cases the large
    message-history fields are released.
    """
    if state.failed or not state.conversation or len(state.conversation) <= 1:
        if detailed:
            logger.debug("Skipping conversation %s (failed or empty)", state.permutation_index)
        counters.failed += 1
        state.turns = []
        state.conversation = []
        state.messages_Patient_assist = []
        state.messages_Therapist_assist = []
        return False

    if state.session_ended_by is not None:
        counters.session_ended += 1
    else:
        counters.completed += 1

    if save_dir:
        save_conversation_csv(state, save_dir)
        if detailed:
            logger.debug("Saved conversation_%s.csv (%d utterances)",
                         state.permutation_index, len(state.conversation))

    # Free the message histories after saving (turns/conversation are kept for the caller).
    state.messages_Patient_assist = []
    state.messages_Therapist_assist = []
    return True


def _cuda_batch_telemetry() -> str:
    """Peak/current reserved CUDA memory for a batch log line, then reset the peak counter.

    Read BEFORE ``empty_cache`` so the peak shows how close the batch came to the device limit.
    Returns "" without CUDA.
    """
    if not torch.cuda.is_available():
        return ""
    peak = torch.cuda.max_memory_reserved() / 1024**3
    cap_txt = ""
    get_frac = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if get_frac is not None:
        frac = get_frac()
        if frac < 1.0:
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cap_txt = f" / cap {frac * total:.1f}G"
    free_dev = torch.cuda.mem_get_info()[0] / 1024**3
    torch.cuda.reset_peak_memory_stats()
    return f", peak {peak:.1f}G{cap_txt}, device free {free_dev:.1f}G"


def _run_one_pass(
    remaining: list[int],
    permutations: list[dict[str, Any]],
    completed: dict[int, ConversationState],
    counters: _Counters,
    *,
    therapist_model: Any,
    therapist_tokenizer: Any,
    client: Any,
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    save_dir: str | os.PathLike | None,
    batch_size: int,
    batch_cooldown_seconds: float,
    total: int,
    start_time: float,
    verbose: bool,
    detailed: bool,
    synthesis_kwargs: dict[str, Any],
) -> tuple[bool, bool]:
    """Run one full pass over the remaining indices.

    Mutates ``completed`` and ``counters`` in place. Returns ``(progress_made, hit_oom)``:
    ``progress_made`` is True if any new conversation was added (drives the
    retry-without-progress limit); ``hit_oom`` is True if a batch hit CUDA OOM, in which case
    the pass stops right after saving that batch's finished conversations so the caller can
    halve the batch size before the next batch allocates at the size that just failed.
    """
    num_batches = (len(remaining) + batch_size - 1) // batch_size
    progress_made = False

    for batch_num, batch_start in enumerate(range(0, len(remaining), batch_size), 1):
        batch_indices = remaining[batch_start: batch_start + batch_size]
        permutations_batch = [(idx, permutations[idx]) for idx in batch_indices]
        batch_time_start = time.time()

        if detailed:
            logger.debug("Batch %d/%d: indices %s", batch_num, num_batches, batch_indices)

        batch_result = run_synthesis_batch(
            permutations_batch=permutations_batch,
            therapist_model=therapist_model,
            therapist_tokenizer=therapist_tokenizer,
            client=client,
            therapist_system_prompt=therapist_system_prompt,
            therapist_init_utterance=therapist_init_utterance,
            **synthesis_kwargs,
        )

        final_states = batch_result["final_states"]
        counters.desynced.extend(batch_result["desynced_indices"])
        batch_elapsed = time.time() - batch_time_start
        # A therapist-generation OOM comes back as error_type "oom" WITH final_states (the
        # conversation that OOMed alone is marked failed, the others are intact).
        hit_oom = batch_result["error_type"] == "oom"

        if final_states is None:
            if verbose:
                logger.info("Batch %d/%d FAILED (%s) - %.1fs",
                            batch_num, num_batches, batch_result["error_type"], batch_elapsed)
            _free_cuda_memory()
            if hit_oom and batch_size > 1:
                return progress_made, True
            time.sleep(batch_cooldown_seconds)
            continue

        batch_saved = 0
        for state in final_states:
            if _record_completed_state(state, save_dir, counters, detailed):
                completed[state.permutation_index] = state
                progress_made = True
                batch_saved += 1

        # Release this batch's KV cache and activations before the next batch allocates its own.
        # Consecutive batches have different maximum sequence lengths, so freed blocks are often
        # the wrong size to reuse and the caching allocator's high-water mark would otherwise
        # grow with every batch (on small GPUs an over-budget request can be fatal rather than a
        # clean OutOfMemoryError). ``empty_cache`` frees only UNUSED cached blocks, so results
        # are unchanged; the cost is one re-allocation per batch.
        del final_states, batch_result
        peak_txt = _cuda_batch_telemetry()
        _free_cuda_memory()

        if verbose:
            total_elapsed = time.time() - start_time
            reserved = (
                f", vram {torch.cuda.memory_reserved() / 1024**3:.1f}G{peak_txt}"
                if torch.cuda.is_available() else ""
            )
            logger.info(
                "Batch %d/%d: %d/%d saved - %d/%d total (%.0f%%) - batch %.1fs, total %.1fs%s",
                batch_num, num_batches, batch_saved, len(batch_indices), len(completed), total,
                len(completed) / total * 100, batch_elapsed, total_elapsed, reserved,
            )

        if hit_oom and batch_size > 1:
            if verbose:
                logger.info("Batch %d/%d hit CUDA OOM - ending this pass so the batch size can "
                            "be halved", batch_num, num_batches)
            return progress_made, True

    return progress_made, False


def _summarize(
    completed: dict[int, ConversationState],
    counters: _Counters,
    total: int,
    elapsed: float,
    detailed: bool,
) -> None:
    """Log the final per-run outcome summary."""
    saved_total = len(completed)
    missing = total - saved_total
    logger.info(
        "Generation summary (%.1fs): requested %d, saved %d (session ended %d, completed %d), "
        "desynced %d, failed/empty %d, missing %d",
        elapsed, total, saved_total, counters.session_ended, counters.completed,
        len(counters.desynced), counters.failed, missing,
    )
    if counters.desynced and detailed:
        logger.debug("Desynced indices: %s", counters.desynced)


def generate_all_conversations(
    therapist_model: Any,
    therapist_tokenizer: Any,
    client: Any,
    permutations: list[dict[str, Any]],
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    save_dir: str | os.PathLike | None = None,
    patient_model_id: str = "gpt-4o-mini-2024-07-18",
    max_tokens_per_response: int = 200,
    num_utterances: int = 49,
    temperature_therapist: float = 0.9,
    temperature_patient: float = 0.7,
    batch_size: int = 8,
    therapist_max_input_tokens: int = 2048,
    patient_api_concurrency: int = 16,
    patient_api_max_retries: int = 3,
    patient_api_seed: int | None = None,
    patient_api_backoff_seconds: float = 1.0,
    batch_cooldown_seconds: float = 1.0,
    max_retries_without_progress: int = 3,
    stop_strings: list[str] | None = None,
    include_empty_init_user_message: bool = False,
    verbose: bool = False,
    verbose_detailed: bool = False,
) -> list[ConversationState]:
    """Generate one conversation per entry of ``permutations``, with retry and OOM recovery.

    Conversation ``i`` uses ``permutations[i]`` and, when ``save_dir`` is given, is written to
    ``save_dir/conversation_<i>.csv`` as soon as it finishes. Existing CSVs are loaded instead
    of regenerated, so an interrupted call resumes where it stopped. Missing indices are retried
    in further passes until all are done or ``max_retries_without_progress`` consecutive passes
    add nothing. A CUDA OOM halves ``batch_size`` for the rest of the call (not counted as a
    retry).

    ``verbose`` enables INFO progress lines; ``verbose_detailed`` enables per-batch/per-turn
    DEBUG lines.

    Returns the finished (non-failed) states, in completion order.
    """
    total = len(permutations)
    completed: dict[int, ConversationState] = {}
    counters = _Counters()
    start_time = time.time()
    detailed = verbose_detailed

    # Bundle the per-synthesis knobs once to keep _run_one_pass's signature readable.
    synthesis_kwargs = dict(
        patient_api_concurrency=patient_api_concurrency,
        include_empty_init_user_message=include_empty_init_user_message,
        num_utterances=num_utterances,
        patient_model_id=patient_model_id,
        max_tokens_per_response=max_tokens_per_response,
        temperature_patient=temperature_patient,
        temperature_therapist=temperature_therapist,
        patient_api_max_retries=patient_api_max_retries,
        patient_api_backoff_seconds=patient_api_backoff_seconds,
        patient_api_seed=patient_api_seed,
        therapist_max_input_tokens=therapist_max_input_tokens,
        stop_strings=stop_strings,
        verbose_detailed=verbose_detailed,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        completed = _resume_from_disk(save_dir, total, verbose=verbose)

    retries_without_progress = 0
    while True:
        remaining = [i for i in range(total) if i not in completed]

        if not remaining:
            if verbose:
                logger.info("All %d conversations completed (%.1fs)",
                            total, time.time() - start_time)
            break

        if retries_without_progress >= max_retries_without_progress:
            logger.warning(
                "Reached retry limit (%d). Stopping with %d conversations missing.",
                max_retries_without_progress, len(remaining),
            )
            break

        if verbose:
            num_batches = (len(remaining) + batch_size - 1) // batch_size
            logger.info(
                "Remaining: %d/%d conversations (%d/%d done, %d batches) [%.1fs elapsed]",
                len(remaining), total, len(completed), total, num_batches,
                time.time() - start_time,
            )

        progress, hit_oom = _run_one_pass(
            remaining, permutations, completed, counters,
            therapist_model=therapist_model,
            therapist_tokenizer=therapist_tokenizer,
            client=client,
            therapist_system_prompt=therapist_system_prompt,
            therapist_init_utterance=therapist_init_utterance,
            save_dir=save_dir,
            batch_size=batch_size,
            batch_cooldown_seconds=batch_cooldown_seconds,
            total=total,
            start_time=start_time,
            verbose=verbose,
            detailed=detailed,
            synthesis_kwargs=synthesis_kwargs,
        )

        # OOM backoff: halve the batch (sticky for the rest of this call) and do not count the
        # pass as a retry, since the configuration changed. Only an OOM at batch size 1 counts
        # toward the retry limit.
        if hit_oom and batch_size > 1:
            batch_size = max(1, batch_size // 2)
            logger.warning("CUDA OOM: conversation batch size halved to %d for the rest of this "
                           "run", batch_size)
            _free_cuda_memory()
            continue

        if progress:
            retries_without_progress = 0
        else:
            retries_without_progress += 1
            if verbose:
                logger.info("No progress this pass. Retry %d/%d",
                            retries_without_progress, max_retries_without_progress)

    _summarize(completed, counters, total, time.time() - start_time, detailed)
    return list(completed.values())
