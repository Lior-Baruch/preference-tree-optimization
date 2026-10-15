"""Oracle scoring: one questionnaire-grading LLM call with strict validation and retries.

The oracle reads a therapist/patient transcript and fills in a questionnaire (Q1, Q2, ...)
through an OpenAI-compatible ``chat.completions`` endpoint with a strict JSON-schema
``response_format``. Only questionnaires whose schema is a flat ``scores`` array (1, 2, 3, 4, 6)
can serve as a training oracle; the nested-schema instruments are evaluation-only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .questionnaires import get_prompt_eval_questionnaire

logger = logging.getLogger(__name__)

__all__ = ["OracleAsyncPrimitives", "OracleConfig", "get_evaluation_json"]


@dataclass(frozen=True)
class OracleConfig:
    """Oracle (questionnaire-grading LLM) settings.

    Attributes:
        model_id: Model name sent to the OpenAI-compatible endpoint.
        request_timeout: Per-request timeout in seconds.
        max_retries: Total attempts per (transcript, questionnaire) call.
        eval_temperature: Sampling temperature of the oracle.
        max_concurrency: Upper bound on in-flight oracle requests; this is the size of
            :meth:`OracleAsyncPrimitives.oracle_sem` when built via
            :meth:`OracleAsyncPrimitives.from_configs`.
    """

    model_id: str
    request_timeout: float = 60.0
    max_retries: int = 3
    eval_temperature: float = 0.0
    max_concurrency: int = 128


class OracleAsyncPrimitives:
    """Semaphores and the GPU lock, created lazily per running event loop.

    asyncio primitives bind to the loop they are first used on and raise ``RuntimeError``
    when reused from another loop. The pipeline runs several short-lived loops (a fresh one
    per generation batch and per tree-building phase, possibly in a worker thread when the
    caller already owns a loop, e.g. Jupyter), so primitives are created on first request
    inside each loop. The cache is keyed by the loop object itself (weakly), not by ``id()``:
    a new loop can never inherit a dead loop's primitives through id reuse, and entries of
    closed loops are discarded.

    One ``gpu_lock`` per loop is shared by branch sampling and look-ahead generation so the
    policy never runs two ``generate`` calls concurrently.
    """

    def __init__(self, oracle_concurrency: int, lookahead_patient_concurrency: int) -> None:
        self._oracle_concurrency = oracle_concurrency
        self._lookahead_patient_concurrency = lookahead_patient_concurrency
        self._by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, Any]] = (
            weakref.WeakKeyDictionary()
        )
        # Fallback for loop implementations that do not support weak references. Holding a
        # strong reference keeps the loop alive, so the identity check below cannot be fooled.
        self._fallback_loop: asyncio.AbstractEventLoop | None = None
        self._fallback: dict[str, Any] = {}

    @classmethod
    def from_configs(
        cls, oracle_cfg: OracleConfig, lookahead_patient_concurrency: int
    ) -> OracleAsyncPrimitives:
        """Build primitives with the oracle semaphore sized by ``oracle_cfg.max_concurrency``."""
        return cls(
            oracle_concurrency=oracle_cfg.max_concurrency,
            lookahead_patient_concurrency=lookahead_patient_concurrency,
        )

    def _loop_cache(self) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        try:
            cache = self._by_loop.get(loop)
        except TypeError:  # loop type without __weakref__
            if self._fallback_loop is not loop:
                self._fallback_loop, self._fallback = loop, {}
            return self._fallback
        if cache is None:
            # A primitive that ever waited holds a strong reference to its loop, which would
            # pin that weak-keyed entry; drop entries of closed loops explicitly.
            for stale in [lp for lp in list(self._by_loop.keys()) if lp.is_closed()]:
                del self._by_loop[stale]
            cache = self._by_loop[loop] = {}
        return cache

    def _get(self, name: str, factory: Callable[[], Any]) -> Any:
        cache = self._loop_cache()
        if name not in cache:
            cache[name] = factory()
        return cache[name]

    def oracle_sem(self) -> asyncio.Semaphore:
        """Bounds concurrent oracle requests."""
        return self._get("oracle_sem", lambda: asyncio.Semaphore(self._oracle_concurrency))

    def gpu_lock(self) -> asyncio.Lock:
        """Serializes policy ``generate`` calls within one loop."""
        return self._get("gpu_lock", asyncio.Lock)

    def lookahead_patient_sem(self) -> asyncio.Semaphore:
        """Bounds concurrent patient-simulator requests issued during tree building."""
        return self._get(
            "lookahead_patient_sem",
            lambda: asyncio.Semaphore(self._lookahead_patient_concurrency),
        )


# Programming errors, not transient failures: retrying cannot help.
_NON_RETRYABLE = (KeyError, TypeError)


async def get_evaluation_json(
    client: Any,
    oracle_cfg: OracleConfig,
    primitives: OracleAsyncPrimitives,
    transcript: str,
    completion: str,
    questionnaire_id: int,
    full_conversation_override: str | None = None,
) -> tuple[dict[str, Any] | None, int, int]:
    """Score one (conversation, questionnaire) pair with the oracle.

    The response must carry the requested ``questionnaire_id``, exactly ``questions_count``
    scores, and every score must be an ``int`` within ``[scale_min, scale_max]``. Invalid or
    failed responses are retried with exponential backoff (1 s, 2 s, ...) up to
    ``oracle_cfg.max_retries`` attempts; ``KeyError`` / ``TypeError`` fail immediately.

    The prompt is sent as a single user message laid out instructions-and-rubric first,
    transcript last, so the shared prefix is eligible for provider-side prompt caching.

    Args:
        client: An ``openai.AsyncOpenAI``-compatible client.
        oracle_cfg: Oracle settings.
        primitives: Loop-local primitives; the oracle semaphore bounds concurrency.
        transcript: Conversation so far, formatted with ``[THERAPIST]:`` / ``[PATIENT]:``.
        completion: Candidate therapist turn appended to ``transcript``.
        questionnaire_id: Questionnaire to fill in (must have a flat ``scores`` schema).
        full_conversation_override: Use this text verbatim instead of
            ``transcript + completion`` (e.g. a look-ahead-extended transcript).

    Returns:
        ``(data, n_questions, attempts)``: ``data`` is the parsed JSON plus ``mean_score``
        (``None`` on failure) and ``attempts`` the number of oracle calls made.
    """
    full_conversation = (
        full_conversation_override
        if full_conversation_override is not None
        else f"{transcript}\n\n[THERAPIST]: {completion}"
    )
    eval_dict = get_prompt_eval_questionnaire(
        questionnaire=questionnaire_id,
        conversation=full_conversation,
    )
    eval_prompt = eval_dict["prompt"]
    n_questions = int(eval_dict["questions_count"])
    schema = eval_dict["schema"]
    scale_min = eval_dict["scale_min"]
    scale_max = eval_dict["scale_max"]

    for attempt in range(oracle_cfg.max_retries):
        try:
            async with primitives.oracle_sem():
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=oracle_cfg.model_id,
                        messages=[{"role": "user", "content": eval_prompt}],
                        temperature=oracle_cfg.eval_temperature,
                        max_tokens=256,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": f"questionnaire_{questionnaire_id}_evaluation",
                                "schema": schema,
                                "strict": True,
                            },
                        },
                    ),
                    timeout=oracle_cfg.request_timeout,
                )
            content = resp.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty oracle response")

            data = json.loads(content)

            if data.get("questionnaire_id") != questionnaire_id:
                raise ValueError(f"Wrong questionnaire_id: {data.get('questionnaire_id')}")

            scores = data.get("scores", [])
            if len(scores) != n_questions:
                raise ValueError(f"Expected {n_questions} scores, got {len(scores)}")
            if any(not isinstance(s, int) or s < scale_min or s > scale_max for s in scores):
                raise ValueError(f"Invalid score values (expected {scale_min}-{scale_max})")

            data["mean_score"] = float(np.mean(scores))
            return data, n_questions, attempt + 1

        except _NON_RETRYABLE as e:
            logger.warning("Oracle non-retryable error (qid=%s): %s", questionnaire_id, e)
            return None, n_questions, attempt + 1

        except Exception as e:
            if attempt >= oracle_cfg.max_retries - 1:
                logger.warning(
                    "Oracle failed after %d attempts (qid=%s): %s",
                    oracle_cfg.max_retries, questionnaire_id, e,
                )
                return None, n_questions, oracle_cfg.max_retries
            await asyncio.sleep(2**attempt)

    return None, n_questions, 0
