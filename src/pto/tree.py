"""Preference-tree construction: branch, look ahead, score, and extract DPO pairs.

At a branch point (the conversation so far ends on a patient turn) the policy samples ``M``
therapist completions. Each one is optionally rolled forward ``K`` turns against the simulated
patient (look-ahead), the resulting transcript is scored by the oracle, and the best and worst
candidates become a (chosen, rejected) pair when their score gap exceeds ``tau``
(:func:`select_pair`).

Two tree modes:

- **greedy** (:func:`grow_preference_trees_batch`, the PTO default): seed each tree with the first
  ``min_conv_length`` utterances of a freshly generated conversation and grow ONE trunk: at every
  therapist turn branch ``M`` ways, append the best candidate to the trunk (so the winner feeds the
  next branch point), let the patient reply, repeat until the trunk reaches its target length.
  All trunks advance in lock-step so each depth is one batched ``generate`` call.
- **independent** (:func:`extract_pref_pairs_from_conversations`): branch at every patient turn of
  an already recorded conversation; the winner is never fed back.

Resume: the pair build is the most expensive phase of an iteration, so progress is persisted under
``iteration_N/pref_pairs/``:

- ``pairs.csv``: completion marker. Present (and its ``pairs.meta.json`` fingerprint matches the
  current config) means the build is done; it is reloaded instead of rebuilt.
- ``_progress.json``: atomic snapshot written after every conversation (independent) or every
  depth (greedy); a restarted build continues from it when mode, iteration, config fingerprint and
  conversation set all match. Deleted once ``pairs.csv`` is written.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import _compat  # noqa: F401
from .conversation import (
    ConversationState,
    _process_session_response,
    build_truncated_training_prompt,
    format_conversation_for_oracle,
    turns_to_messages,
    turns_to_patient_messages,
)
from .generation import generate_patient_responses_batch, generate_therapist_responses_batch
from .lookahead import LookaheadConfig, _count_role_labels, simulate_lookahead_batch
from .oracle import OracleAsyncPrimitives, OracleConfig, get_evaluation_json

if TYPE_CHECKING:
    from .config import PTORunConfig

logger = logging.getLogger(__name__)

__all__ = [
    "PAIRS_FILENAME",
    "PAIRS_META_FILENAME",
    "PROGRESS_FILENAME",
    "build_pref_pairs_for_conversation",
    "extract_pref_pairs_from_conversations",
    "grow_preference_trees_batch",
    "load_completed_pairs",
    "save_completed_pairs",
    "select_pair",
]

PAIRS_FILENAME = "pairs.csv"
PAIRS_META_FILENAME = "pairs.meta.json"
PROGRESS_FILENAME = "_progress.json"


# ── Pair selection ───────────────────────────────────────────────────────────────


def _rank_scores(scores: Sequence[float | None]) -> list[tuple[float, int]]:
    """``(score, index)`` of every scored candidate, ascending by score.

    The sort is stable, so among tied scores the lower index ranks lower: the best candidate is
    the LAST index among the maxima and the worst is the FIRST index among the minima.
    """
    ranked = [(s, i) for i, s in enumerate(scores) if s is not None]
    ranked.sort(key=lambda si: si[0])
    return ranked


def select_pair(scores: Sequence[float | None], tau: float) -> tuple[int, int] | None:
    """Pick ``(best_idx, worst_idx)`` for a DPO pair, or ``None`` when no pair qualifies.

    ``None`` entries (oracle failure, or an empty completion masked out by the caller) are
    ignored. A pair needs at least two scored candidates and a STRICT gap ``best - worst > tau``,
    so all-tied candidates never form a pair even with ``tau = 0``.
    """
    ranked = _rank_scores(scores)
    if len(ranked) >= 2 and (ranked[-1][0] - ranked[0][0]) > tau:
        return ranked[-1][1], ranked[0][1]
    return None


# ── Branch scoring ───────────────────────────────────────────────────────────────


def _sample_M_completions(  # M = branch count, as in the paper
    policy: Any,
    tokenizer: Any,
    prefix_messages: list[dict[str, str]],
    num_branches: int,
    temperature: float,
    max_tokens: int,
    max_input_tokens: int,
    stop_strings: list[str] | None,
) -> list[str]:
    """Sample ``num_branches`` therapist completions from one prefix (independent mode).

    The prefix is repeated ``num_branches`` times in one sampled batch, so draws are independent.
    Empty / whitespace-only completions are dropped.
    """
    responses, error = generate_therapist_responses_batch(
        therapist_model=policy,
        therapist_tokenizer=tokenizer,
        batch_messages=[prefix_messages] * num_branches,
        max_tokens=max_tokens,
        temperature=temperature,
        max_input_tokens=max_input_tokens,
        stop_strings=stop_strings,
    )
    if error is not None or responses is None:
        logger.warning("Branch sampling failed (%s); skipping this branch point", error)
        return []
    return [r for r in responses if r and r.strip()]


async def _oracle_score_extended_detailed(
    client: Any,
    oracle_cfg: OracleConfig,
    primitives: OracleAsyncPrimitives,
    full_conversation: str,
    questionnaire_ids: Sequence[int],
) -> dict[str, Any]:
    """Oracle-score one already-extended transcript.

    Returns ``{"score", "sub_scores", "success", "retries"}``: ``score`` is the mean over
    ``questionnaire_ids`` (``None`` if any questionnaire fails), ``sub_scores`` maps each id (as a
    string) to its mean, ``retries`` counts oracle calls beyond the first per questionnaire.
    """
    rewards: list[float] = []
    sub_scores: dict[str, float] = {}
    total_attempts = 0
    num_calls = 0
    for qid in questionnaire_ids:
        data, _, attempts = await get_evaluation_json(
            client,
            oracle_cfg,
            primitives,
            transcript="",
            completion="",
            questionnaire_id=int(qid),
            full_conversation_override=full_conversation,
        )
        num_calls += 1
        total_attempts += attempts
        if data is None:
            return {
                "score": None,
                "sub_scores": (sub_scores or None),
                "success": False,
                "retries": max(0, total_attempts - num_calls),
            }
        m = float(data["mean_score"])
        sub_scores[str(int(qid))] = m
        rewards.append(m)
    return {
        "score": float(np.mean(rewards)),
        "sub_scores": sub_scores,
        "success": True,
        "retries": max(0, total_attempts - num_calls),
    }


async def _score_completions_batch_detailed(
    client: Any,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    *,
    policy: Any,
    tokenizer: Any,
    therapist_system_prompt: str,
    transcripts: list[str],
    completions: list[str],
    patient_system_prompts: list[str],
    questionnaire_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Look-ahead (when ``k > 0``) + oracle-score parallel (transcript, completion) items.

    Returns one detail dict per item: ``score`` / ``sub_scores`` / ``success`` / ``retries`` plus
    ``lookahead = {k, realized_turns, ended_early, tail}`` where ``tail`` is only the simulated
    continuation (the prefix + completion seed is sliced off; look-ahead concatenates, so the
    slice is exact).
    """
    if lookahead_cfg.k > 0:
        full_convs = await simulate_lookahead_batch(
            transcripts=transcripts,
            completions=completions,
            system_prompt_therapist=therapist_system_prompt,
            system_prompts_patient=patient_system_prompts,
            therapist_model=policy,
            therapist_tokenizer=tokenizer,
            client=client,
            patient_model_id=lookahead_cfg.patient_model_id,
            lookahead_k=lookahead_cfg.k,
            temperature_therapist=lookahead_cfg.temperature_therapist,
            temperature_patient=lookahead_cfg.temperature_patient,
            max_tokens=lookahead_cfg.max_tokens,
            max_input_tokens=lookahead_cfg.max_input_tokens,
            stop_strings=lookahead_cfg.stop_strings,
            patient_sem=primitives.lookahead_patient_sem(),
            gpu_lock=primitives.gpu_lock(),
            patient_max_retries=lookahead_cfg.patient_api_max_retries,
            patient_backoff_seconds=lookahead_cfg.patient_api_backoff_seconds,
            sub_batch_size=lookahead_cfg.lookahead_sub_batch_size,
        )
    else:
        full_convs = [f"{t}\n\n[THERAPIST]: {c}" for t, c in zip(transcripts, completions)]

    details = await asyncio.gather(
        *[
            _oracle_score_extended_detailed(client, oracle_cfg, primitives, fc, questionnaire_ids)
            for fc in full_convs
        ]
    )

    out: list[dict[str, Any]] = []
    for i, d in enumerate(details):
        if lookahead_cfg.k > 0:
            realized = max(
                0, _count_role_labels(full_convs[i]) - _count_role_labels(transcripts[i]) - 1
            )
            seed = f"{transcripts[i]}\n\n[THERAPIST]: {completions[i]}"
            ext = full_convs[i]
            tail = ext[len(seed):] if ext.startswith(seed) else ext
            la = {
                "k": lookahead_cfg.k,
                "realized_turns": realized,
                "ended_early": realized < lookahead_cfg.k,
                "tail": tail,
            }
        else:
            la = {"k": 0, "realized_turns": 0, "ended_early": False, "tail": None}
        out.append({**d, "lookahead": la})
    return out


async def _score_branches_detailed(
    client: Any,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    *,
    policy: Any,
    tokenizer: Any,
    therapist_system_prompt: str,
    patient_system_prompt: str,
    transcript: str,
    completions: list[str],
    questionnaire_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Score ``len(completions)`` candidates that share ONE prefix (independent mode)."""
    n = len(completions)
    return await _score_completions_batch_detailed(
        client,
        oracle_cfg,
        lookahead_cfg,
        primitives,
        policy=policy,
        tokenizer=tokenizer,
        therapist_system_prompt=therapist_system_prompt,
        transcripts=[transcript] * n,
        completions=completions,
        patient_system_prompts=[patient_system_prompt] * n,
        questionnaire_ids=questionnaire_ids,
    )


def _record_pto_branch(
    recorder: Any,
    *,
    iteration: int,
    phase: str,
    conversation_id: int,
    branch_id: int,
    prefix: str,
    completions: list[str],
    details: list[dict[str, Any]],
    best_idx: int,
    worst_idx: int,
    emitted_pair: bool,
) -> None:
    """Append ONE branch record (prefix once, all ``M`` candidates nested) to ``recorder``.

    ``details`` aligns with ``completions``. A candidate's role is ``chosen`` for the top-ranked
    one (always present: greedy mode always appends it), ``rejected`` only when a pair was
    emitted, else ``neither``. Candidates whose scoring failed are still recorded. No-op when the
    recorder is missing or disabled.
    """
    if recorder is None or not getattr(recorder, "enabled", False):
        return
    cands = []
    for idx, (c, d) in enumerate(zip(completions, details)):
        if idx == best_idx:
            role = "chosen"
        elif idx == worst_idx and emitted_pair:
            role = "rejected"
        else:
            role = "neither"
        cands.append(
            {
                "idx": int(idx),
                "completion": c,
                "score": d.get("score"),
                "sub_scores": d.get("sub_scores"),
                "role": role,
                "oracle": {
                    "success": bool(d.get("success", False)),
                    "retries": int(d.get("retries", 0)),
                },
                "lookahead": d.get("lookahead"),
            }
        )
    recorder.append(
        {
            "method": "PTO",
            "iteration": int(iteration),
            "phase": phase,
            "conversation_id": int(conversation_id),
            "branch_id": int(branch_id),
            "prefix": prefix,
            "chosen_idx": (int(best_idx) if best_idx is not None and best_idx >= 0 else None),
            "candidates": cands,
        }
    )


# ── Persistence: pairs.csv (+ fingerprint) and _progress.json ────────────────────


def _json_default(o: Any) -> Any:
    """``json.dumps`` fallback for numpy scalars/arrays that may sit in EDA records."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _atomic_write_text(path: str, text: str) -> None:
    """Write text atomically (tmp + ``os.replace``) so a crash never leaves a torn file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _write_pairs_csv(pref_pairs: list[dict[str, Any]], path: str) -> None:
    """Atomically write the preference pairs (the build's completion marker)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    pd.DataFrame(pref_pairs).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _reload_pairs_csv(path: str) -> list[dict[str, Any]]:
    """Rebuild the ``pref_pairs`` list from a completed ``pairs.csv``.

    ``keep_default_na=False`` keeps prompt/chosen/rejected as literal strings (no NaN coercion of
    empty fields); the two score columns are coerced back to float. An empty CSV (a 0-pair
    build) returns ``[]`` so the caller's zero-pairs guard fires exactly as on a fresh build.
    """
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return []
    records = df.to_dict("records")
    for r in records:
        for k in ("chosen_score", "rejected_score"):
            if k in r and r[k] != "":
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    pass
    return records


def _pref_config_key(cfg: PTORunConfig) -> dict[str, Any]:
    """Fingerprint of the main settings that shape a build's pairs: tree mode, MCL, M, tau,
    conversation length, greedy trunk target, look-ahead K, questionnaires, oracle model and seed.

    A resumed ``_progress.json`` and a reused ``pairs.csv`` must match it; the run name may be
    user-chosen, so it cannot be relied on to separate runs with different settings. Sampling
    settings (temperatures, max_new_tokens), ``dpo.max_prompt_length`` and the policy/patient
    models are not included; delete ``pref_pairs/`` to rebuild after changing them.
    """
    return {
        "mode": cfg.tree.mode,
        "min_conv_length": cfg.tree.min_conv_length,
        "num_branches_per_turn": cfg.tree.branches,
        "pref_filter_tau": cfg.tree.tau,
        "num_utterances_for_data": cfg.loop.conversation_length,
        "greedy_trunk_target_len": cfg.tree.trunk_target_len,
        "lookahead_k": cfg.lookahead.k,
        "questionnaire_ids": [int(q) for q in cfg.oracle.questionnaires],
        "oracle_model": cfg.models.oracle,
        "seed": cfg.seed,
    }


def _write_pref_progress(
    progress_path: str,
    *,
    mode: str,
    iteration: int,
    cfg: PTORunConfig,
    perm_ids: Sequence[int],
    pairs: list[dict[str, Any]],
    eda_records: list[dict[str, Any]],
    depth: int = 0,
    processed_conv_ids: Sequence[int] | None = None,
    trunks: list[dict[str, Any]] | None = None,
) -> None:
    """Atomically snapshot the in-progress build for resume."""
    snap = {
        "mode": mode,
        "iteration": int(iteration),
        "config_key": _pref_config_key(cfg),
        "perm_ids": sorted(int(p) for p in perm_ids),
        "depth": int(depth),
        "processed_conv_ids": sorted(int(c) for c in (processed_conv_ids or [])),
        "trunks": trunks or [],
        "pairs": pairs,
        "eda_records": eda_records or [],
    }
    _atomic_write_text(progress_path, json.dumps(snap, ensure_ascii=False, default=_json_default))


def _load_pref_progress(
    progress_path: str | None,
    cfg: PTORunConfig,
    iteration: int,
    mode: str,
    expected_perm_ids: Sequence[int],
) -> dict[str, Any] | None:
    """Load a progress snapshot iff it matches the current build, else ``None`` (rebuild).

    Guards on mode, iteration, config fingerprint and the conversation set, and treats a
    corrupt or partial file as absent.
    """
    if not progress_path or not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, encoding="utf-8") as f:
            snap = json.load(f)
    except Exception as e:  # any unreadable snapshot means "rebuild"
        logger.warning("%s unreadable (%s: %s); rebuilding from scratch", progress_path,
                       type(e).__name__, e)
        return None
    reasons = []
    if snap.get("mode") != mode:
        reasons.append("mode")
    if snap.get("iteration") != int(iteration):
        reasons.append("iteration")
    if snap.get("config_key") != _pref_config_key(cfg):
        reasons.append("config")
    if snap.get("perm_ids") != sorted(int(p) for p in expected_perm_ids):
        reasons.append("perm_ids")
    if reasons:
        logger.warning(
            "Stale %s (mismatch: %s); rebuilding from scratch", PROGRESS_FILENAME,
            ", ".join(reasons),
        )
        return None
    return snap


def _remove_progress(progress_path: str) -> None:
    """Drop the in-build snapshot once ``pairs.csv`` is the completion marker."""
    if os.path.exists(progress_path):
        try:
            os.remove(progress_path)
        except OSError:
            pass


def load_completed_pairs(pref_dir: str, cfg: PTORunConfig) -> list[dict[str, Any]] | None:
    """Reload a completed build from ``pref_dir/pairs.csv``, or ``None`` if it must be (re)built.

    A ``pairs.csv`` is reused only when the sidecar ``pairs.meta.json`` fingerprint equals the
    current :func:`_pref_config_key`. Otherwise a warning is logged and the stale files are
    removed, so a crash during the rebuild can never pair the new fingerprint with old pairs.
    """
    pairs_csv = os.path.join(pref_dir, PAIRS_FILENAME)
    meta_path = os.path.join(pref_dir, PAIRS_META_FILENAME)
    if not os.path.exists(pairs_csv):
        return None
    saved_key = None
    try:
        with open(meta_path, encoding="utf-8") as f:
            saved_key = json.load(f).get("config_key")
    except (OSError, ValueError, AttributeError):
        saved_key = None
    current_key = json.loads(json.dumps(_pref_config_key(cfg)))
    if saved_key != current_key:
        reason = "missing fingerprint" if saved_key is None else "config changed"
        logger.warning(
            "Ignoring stale %s (%s); rebuilding the preference pairs", pairs_csv, reason
        )
        for p in (pairs_csv, meta_path):
            try:
                os.remove(p)
            except OSError:
                pass
        return None
    return _reload_pairs_csv(pairs_csv)


def save_completed_pairs(
    pref_pairs: list[dict[str, Any]], pref_dir: str, cfg: PTORunConfig
) -> str:
    """Write the fingerprint, then ``pairs.csv`` (the marker, written last). Returns its path."""
    pairs_csv = os.path.join(pref_dir, PAIRS_FILENAME)
    meta = {"config_key": _pref_config_key(cfg), "num_pairs": len(pref_pairs)}
    _atomic_write_text(
        os.path.join(pref_dir, PAIRS_META_FILENAME), json.dumps(meta, indent=2)
    )
    _write_pairs_csv(pref_pairs, pairs_csv)
    return pairs_csv


# ── Independent mode ─────────────────────────────────────────────────────────────


async def build_pref_pairs_for_conversation(
    state: ConversationState,
    permutation: dict[str, Any],
    *,
    policy: Any,
    tokenizer: Any,
    client: Any,
    therapist_system_prompt: str,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    cfg: PTORunConfig,
    recorder: Any = None,
    iteration: int = 0,
) -> list[dict[str, Any]]:
    """All (prompt, chosen, rejected) pairs from one recorded conversation.

    Branch points are patient turns ``i`` with at least ``min_conv_length`` utterances so far
    (``i + 1 >= MCL``) that are followed by a recorded therapist turn.
    """
    turns = state.turns
    if not turns:
        return []

    tc = cfg.tree
    patient_system_prompt = permutation["patient_system_prompt"]
    pairs: list[dict[str, Any]] = []

    for i, turn in enumerate(turns):
        if turn["role"] != "patient":
            continue
        if (i + 1) < tc.min_conv_length:
            continue
        # The final patient turn has no recorded therapist reply to anchor a pair.
        if i + 1 >= len(turns):
            continue

        partial_turns = turns[: i + 1]  # ends on a patient turn
        prefix_messages = turns_to_messages(partial_turns, therapist_system_prompt)
        transcript = format_conversation_for_oracle(prefix_messages)

        # Render the prompt exactly as DPOTrainer will see it, capped by dropping the oldest
        # turns. Skip the branch point before the expensive sampling if even the most recent turn
        # does not fit.
        prompt = build_truncated_training_prompt(
            partial_turns,
            therapist_system_prompt,
            tokenizer,
            max_prompt_tokens=cfg.dpo.max_prompt_length,
        )
        if prompt is None:
            continue

        completions = _sample_M_completions(
            policy,
            tokenizer,
            prefix_messages=prefix_messages,
            num_branches=tc.branches,
            temperature=tc.temperature,
            max_tokens=tc.max_new_tokens,
            max_input_tokens=cfg.generation.max_input_tokens,
            stop_strings=cfg.generation.stop_strings,
        )
        if len(completions) < 2:
            continue  # a pair needs two candidates

        branch_details = await _score_branches_detailed(
            client,
            oracle_cfg,
            lookahead_cfg,
            primitives,
            policy=policy,
            tokenizer=tokenizer,
            therapist_system_prompt=therapist_system_prompt,
            patient_system_prompt=patient_system_prompt,
            transcript=transcript,
            completions=completions,
            questionnaire_ids=cfg.oracle.questionnaires,
        )

        scores = [d["score"] for d in branch_details]
        ranked = _rank_scores(scores)
        best_idx = ranked[-1][1] if ranked else -1
        worst_idx = ranked[0][1] if ranked else -1
        pair = select_pair(scores, tc.tau)

        _record_pto_branch(
            recorder,
            iteration=iteration,
            phase="independent",
            conversation_id=state.permutation_index,
            branch_id=i,
            prefix=transcript,
            completions=completions,
            details=branch_details,
            best_idx=best_idx,
            worst_idx=worst_idx,
            emitted_pair=pair is not None,
        )

        if pair is None:
            continue
        best_k, worst_k = pair
        pairs.append(
            {
                "prompt": prompt,
                "chosen": completions[best_k],
                "rejected": completions[worst_k],
                "chosen_score": scores[best_k],
                "rejected_score": scores[worst_k],
                "conversation_id": state.permutation_index,
                "branch_turn_index": i,
            }
        )

    return pairs


async def extract_pref_pairs_from_conversations(
    completed_states: list[ConversationState],
    permutations: list[dict[str, Any]],
    *,
    policy: Any,
    tokenizer: Any,
    client: Any,
    therapist_system_prompt: str,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    cfg: PTORunConfig,
    recorder: Any = None,
    iteration: int = 0,
    progress_path: str | None = None,
) -> list[dict[str, Any]]:
    """Independent-mode pairs across all conversations.

    Conversations are processed sequentially (branch sampling is GPU-bound); scoring inside a
    conversation is batched. With ``progress_path`` a snapshot is written after each
    conversation, so an interrupted build skips finished conversations on restart.
    """

    def usable(s: ConversationState) -> bool:
        return not (s.failed or not s.conversation or len(s.conversation) <= 1)

    conv_perm_ids = sorted(int(s.permutation_index) for s in completed_states if usable(s))
    recording = recorder is not None and getattr(recorder, "enabled", False)

    all_pairs: list[dict[str, Any]] = []
    processed: set[int] = set()
    resume_state = _load_pref_progress(
        progress_path, cfg, iteration, "independent", conv_perm_ids
    )
    if resume_state is not None:
        all_pairs = list(resume_state.get("pairs", []))
        processed = {int(c) for c in resume_state.get("processed_conv_ids", [])}
        if recording:
            recorder.records = list(resume_state.get("eda_records", []))
        logger.info(
            "Resuming independent build: %d conversations done, %d pairs carried",
            len(processed),
            len(all_pairs),
        )

    for state in completed_states:
        if not usable(state):
            continue
        if int(state.permutation_index) in processed:
            continue
        pairs = await build_pref_pairs_for_conversation(
            state,
            permutations[state.permutation_index],
            policy=policy,
            tokenizer=tokenizer,
            client=client,
            therapist_system_prompt=therapist_system_prompt,
            oracle_cfg=oracle_cfg,
            lookahead_cfg=lookahead_cfg,
            primitives=primitives,
            cfg=cfg,
            recorder=recorder,
            iteration=iteration,
        )
        all_pairs.extend(pairs)
        processed.add(int(state.permutation_index))
        if progress_path is not None:
            _write_pref_progress(
                progress_path,
                mode="independent",
                iteration=iteration,
                cfg=cfg,
                perm_ids=conv_perm_ids,
                pairs=all_pairs,
                eda_records=(recorder.records if recording else []),
                processed_conv_ids=sorted(processed),
            )
        logger.info("Conversation %d: %d pair(s)", state.permutation_index, len(pairs))
    return all_pairs


# ── Greedy mode (lock-step trunk growth) ─────────────────────────────────────────


@dataclass
class _TreeState:
    """One live trunk during greedy growth.

    ``conv`` is a plain :class:`ConversationState`, so the patient- and therapist-perspective
    message lists stay in sync as the trunk grows (via ``_process_session_response``).
    """

    conv: ConversationState
    patient_system_prompt: str
    pairs: list[dict[str, Any]] = field(default_factory=list)


def _slice_prefix_seeds(
    completed_states: list[ConversationState], min_conv_length: int
) -> list[ConversationState]:
    """Seed greedy trees with the first ``min_conv_length`` utterances of each conversation.

    No separate prefix-generation pass is needed. A conversation qualifies only if it extends
    PAST ``min_conv_length`` (so the prefix never ends on a session-terminating patient turn).
    With an even ``min_conv_length`` (enforced by config validation) ``turns[MCL - 1]`` is a
    patient turn and the therapist speaks next.

    Containers are copied (turn dicts, fresh lists) so growing the trunk never mutates the
    original conversation, which is the saved evaluation data for the previous model state.
    Message lists are left empty; the grower rebuilds them from ``turns``.
    """
    seeds: list[ConversationState] = []
    for s in completed_states:
        if s.failed or len(s.turns) <= min_conv_length:
            continue
        prefix_turns = s.turns[:min_conv_length]
        if prefix_turns[-1]["role"] != "patient":
            logger.warning(
                "Conversation %d: MCL prefix does not end on a patient turn (last role=%r); "
                "skipped as a seed",
                s.permutation_index,
                prefix_turns[-1]["role"],
            )
            continue
        seeds.append(
            ConversationState(
                permutation_index=s.permutation_index,
                conversation=list(s.conversation[:min_conv_length]),
                messages_Patient_assist=[],
                messages_Therapist_assist=[],
                turns=[dict(t) for t in prefix_turns],
                next_speaker="therapist",
            )
        )
    return seeds


async def _grow_therapist_depth(
    active: list[_TreeState],
    *,
    client: Any,
    policy: Any,
    tokenizer: Any,
    therapist_system_prompt: str,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    cfg: PTORunConfig,
    M: int,  # branch count, as in the paper
    recorder: Any = None,
    iteration: int = 0,
) -> None:
    """One therapist (branching) depth across all active trunks.

    Branch ``M`` completions per trunk, look-ahead + oracle-score them all in one batch, then per
    trunk append the best completion (always, to advance the trunk) and record a pair when the
    score gap exceeds ``tau``.
    """
    tc = cfg.tree
    # 1. Flatten (trunk x M); the oracle transcript is computed once per trunk.
    transcripts = [
        format_conversation_for_oracle(t.conv.messages_Therapist_assist) for t in active
    ]
    flat_messages = [t.conv.messages_Therapist_assist for t in active for _ in range(M)]

    # 2. Branch-sample in chunks of generation.batch_size to bound memory (active x M can be
    #    large; the look-ahead sub-batches its own therapist turns).
    completions: list[str | None] = [None] * len(flat_messages)
    chunk = max(1, cfg.generation.batch_size)
    gpu_lock = primitives.gpu_lock()
    for start in range(0, len(flat_messages), chunk):
        sub = flat_messages[start : start + chunk]
        async with gpu_lock:
            responses, error = generate_therapist_responses_batch(
                therapist_model=policy,
                therapist_tokenizer=tokenizer,
                batch_messages=sub,
                max_tokens=tc.max_new_tokens,
                temperature=tc.temperature,
                max_input_tokens=cfg.generation.max_input_tokens,
                stop_strings=cfg.generation.stop_strings,
            )
        if error is not None or responses is None:
            logger.warning("Branch sampling failed for a chunk (%s); those branches dropped", error)
            continue
        for j, r in enumerate(responses):
            completions[start + j] = r

    # 3. Look-ahead + oracle-score every non-empty completion in ONE batch.
    score_idx = [i for i, c in enumerate(completions) if c and c.strip()]
    if score_idx:
        packed = await _score_completions_batch_detailed(
            client,
            oracle_cfg,
            lookahead_cfg,
            primitives,
            policy=policy,
            tokenizer=tokenizer,
            therapist_system_prompt=therapist_system_prompt,
            transcripts=[transcripts[i // M] for i in score_idx],
            completions=[completions[i] for i in score_idx],
            patient_system_prompts=[active[i // M].patient_system_prompt for i in score_idx],
            questionnaire_ids=cfg.oracle.questionnaires,
        )
        detail_by_flat = {i: packed[k] for k, i in enumerate(score_idx)}
    else:
        detail_by_flat = {}

    # 4. Per trunk: rank, append the winner, emit a pair if the gap passes tau.
    for ti, t in enumerate(active):
        # Keep all M slots aligned (failed samples / failed oracle calls included) so the EDA
        # record lists every branch.
        trunk_completions: list[str] = []
        trunk_details: list[dict[str, Any]] = []
        for m in range(M):
            flat_i = ti * M + m
            c = completions[flat_i]
            d = detail_by_flat.get(flat_i)
            trunk_completions.append(c if (c and c.strip()) else "")
            trunk_details.append(
                d
                if d is not None
                else {
                    "score": None,
                    "sub_scores": None,
                    "success": False,
                    "retries": 0,
                    "lookahead": None,
                }
            )

        # Empty completion slots are excluded from ranking.
        scores = [
            trunk_details[m]["score"] if trunk_completions[m] else None for m in range(M)
        ]
        ranked = _rank_scores(scores)
        if not ranked:
            t.conv.is_active = False  # freeze: no valid branch to advance the trunk
            continue
        worst_score, worst_m = ranked[0]
        best_score, best_m = ranked[-1]
        best_text = trunk_completions[best_m]
        worst_text = trunk_completions[worst_m]

        # Snapshot the (capped) prompt BEFORE appending the winner. The trunk advances either
        # way; only pair emission needs a renderable prompt (None = even the most recent turn
        # exceeds the budget).
        prompt = build_truncated_training_prompt(
            t.conv.turns,
            therapist_system_prompt,
            tokenizer,
            max_prompt_tokens=cfg.dpo.max_prompt_length,
        )
        branch_depth = len(t.conv.conversation)
        emitted_pair = select_pair(scores, tc.tau) is not None and prompt is not None

        _record_pto_branch(
            recorder,
            iteration=iteration,
            phase="tree",
            conversation_id=t.conv.permutation_index,
            branch_id=branch_depth,
            prefix=transcripts[ti],
            completions=trunk_completions,
            details=trunk_details,
            best_idx=best_m,
            worst_idx=worst_m,
            emitted_pair=emitted_pair,
        )

        # Always append the winner: this is the greedy feedback the independent mode lacks.
        # Flips next_speaker to the patient; a SESSION ENDED inside the winner freezes the trunk
        # (its cleaned text is kept).
        _process_session_response(t.conv, best_text, "therapist", "user", "assistant")

        if emitted_pair:
            t.pairs.append(
                {
                    "prompt": prompt,
                    "chosen": best_text,
                    "rejected": worst_text,
                    "chosen_score": best_score,
                    "rejected_score": worst_score,
                    "conversation_id": t.conv.permutation_index,
                    "branch_depth": branch_depth,
                }
            )


async def _grow_patient_depth(
    active: list[_TreeState],
    *,
    client: Any,
    cfg: PTORunConfig,
    primitives: OracleAsyncPrimitives,
) -> None:
    """One patient depth across all active trunks (one fresh API call per trunk)."""
    g = cfg.generation
    responses = await generate_patient_responses_batch(
        client,
        cfg.models.patient,
        [t.conv.messages_Patient_assist for t in active],
        g.max_new_tokens,
        g.patient_temperature,
        primitives.lookahead_patient_sem(),
        g.patient_max_retries,
        g.patient_backoff_s,
        seed=None,
    )
    for t, resp in zip(active, responses):
        if isinstance(resp, BaseException) or resp is None:
            t.conv.is_active = False
            t.conv.failed = True
            continue
        _process_session_response(t.conv, resp, "patient", "assistant", "user")


async def grow_preference_trees_batch(
    seed_states: list[ConversationState],
    permutations: list[dict[str, Any]],
    *,
    policy: Any,
    tokenizer: Any,
    client: Any,
    therapist_system_prompt: str,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    cfg: PTORunConfig,
    recorder: Any = None,
    iteration: int = 0,
    progress_path: str | None = None,
) -> list[dict[str, Any]]:
    """Greedy preference-tree growth, lock-step across all seed trunks.

    A therapist depth branches ``M`` ways per trunk, scores, and appends the best candidate
    (recording a pair when the gap exceeds ``tau``); a patient depth continues every trunk.
    Trunks grow until they reach :attr:`PTORunConfig.effective_trunk_target_len` utterances or
    freeze (SESSION ENDED, patient API failure, or no scorable branch).

    Returns the flat list of pair dicts (same shape as the independent mode).
    """
    policy.eval()
    policy.config.use_cache = True

    trees: list[_TreeState] = []
    for s in seed_states:
        # A usable seed ends on a patient turn (therapist next) with real content.
        if s.failed or len(s.conversation) < 2 or s.next_speaker != "therapist":
            continue
        patient_sys = permutations[s.permutation_index]["patient_system_prompt"]
        # Rebuild both message views from the intact turns so every depth sees the full system
        # prompt + history.
        s.messages_Therapist_assist = turns_to_messages(s.turns, therapist_system_prompt)
        s.messages_Patient_assist = turns_to_patient_messages(s.turns, patient_sys)
        trees.append(_TreeState(conv=s, patient_system_prompt=patient_sys))
    if not trees:
        return []

    seed_perm_ids = sorted(int(s.permutation_index) for s in seed_states)
    recording = recorder is not None and getattr(recorder, "enabled", False)
    carried_pairs: list[dict[str, Any]] = []
    start_depth = 0
    resume_state = _load_pref_progress(progress_path, cfg, iteration, "greedy", seed_perm_ids)
    if resume_state is not None:
        saved = {int(t["permutation_index"]): t for t in resume_state.get("trunks", [])}
        for tree in trees:
            st = saved.get(int(tree.conv.permutation_index))
            if st is None:
                continue
            tree.conv.turns = [dict(t) for t in st["turns"]]
            tree.conv.conversation = [t["content"] for t in tree.conv.turns]
            tree.conv.next_speaker = st["next_speaker"]
            tree.conv.is_active = bool(st["is_active"])
            tree.conv.messages_Therapist_assist = turns_to_messages(
                tree.conv.turns, therapist_system_prompt
            )
            tree.conv.messages_Patient_assist = turns_to_patient_messages(
                tree.conv.turns, tree.patient_system_prompt
            )
        carried_pairs = list(resume_state.get("pairs", []))
        start_depth = int(resume_state.get("depth", 0))
        if recording:
            recorder.records = list(resume_state.get("eda_records", []))
        n_resumed = sum(1 for t in trees if int(t.conv.permutation_index) in saved)
        logger.info(
            "Resuming greedy build: %d trunks restored at depth %d, %d pairs carried",
            n_resumed,
            start_depth,
            len(carried_pairs),
        )

    M = cfg.tree.branches  # branch count, as in the paper
    target_len = cfg.effective_trunk_target_len
    depth = start_depth

    while True:
        active = [
            t for t in trees if t.conv.is_active and len(t.conv.conversation) < target_len
        ]
        if not active:
            break

        speaker = active[0].conv.next_speaker
        # Defensive desync guard: freeze any trunk that drifted off the shared cadence.
        desynced = [t for t in active if t.conv.next_speaker != speaker]
        if desynced:
            for t in desynced:
                t.conv.is_active = False
            active = [t for t in active if t.conv.is_active]
            if not active:
                break

        if speaker == "therapist":
            await _grow_therapist_depth(
                active,
                client=client,
                policy=policy,
                tokenizer=tokenizer,
                therapist_system_prompt=therapist_system_prompt,
                oracle_cfg=oracle_cfg,
                lookahead_cfg=lookahead_cfg,
                primitives=primitives,
                cfg=cfg,
                M=M,
                recorder=recorder,
                iteration=iteration,
            )
        else:
            await _grow_patient_depth(active, client=client, cfg=cfg, primitives=primitives)

        depth += 1

        # Checkpoint at this lock-step boundary (trunk text + pairs; cheap, atomic).
        if progress_path is not None:
            _write_pref_progress(
                progress_path,
                mode="greedy",
                iteration=iteration,
                cfg=cfg,
                perm_ids=seed_perm_ids,
                pairs=carried_pairs + [p for t in trees for p in t.pairs],
                eda_records=(recorder.records if recording else []),
                depth=depth,
                trunks=[
                    {
                        "permutation_index": t.conv.permutation_index,
                        "turns": t.conv.turns,
                        "next_speaker": t.conv.next_speaker,
                        "is_active": t.conv.is_active,
                    }
                    for t in trees
                ],
            )

        n_pairs = len(carried_pairs) + sum(len(t.pairs) for t in trees)
        logger.info(
            "Tree depth %d (%s): %d active trunks, %d pairs so far",
            depth,
            speaker,
            len(active),
            n_pairs,
        )

    return carried_pairs + [p for t in trees for p in t.pairs]
