"""LLM judges: one structured-JSON call per (conversation, questionnaire).

Two providers are supported:

* ``openai`` -- the training judge (``gpt-4o-mini-2024-07-18``, temperature 0.1, seed 42),
  called with a *strict* ``response_format`` JSON schema. Any OpenAI-compatible endpoint can be
  used through :attr:`JudgeSpec.base_url`.
* ``anthropic`` -- the held-out judge (``claude-haiku-4-5``), called with
  ``output_config.format`` JSON-schema structured output. Claude's JSON schema does not accept
  numeric bounds or array-length constraints, so they are removed before sending and folded into
  the field descriptions (see :func:`_strip_unsupported_constraints`); the response is still
  length- and range-validated client-side by ``questionnaires.parse_json_response``.

The prompt is always a single user message with the rubric first and the transcript last, as
built by ``questionnaires.get_prompt_eval_questionnaire``; nothing here alters it, so scores stay
comparable with the reported numbers.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

PROVIDERS = ("openai", "anthropic")
DEFAULT_OPENAI_JUDGE = "gpt-4o-mini-2024-07-18"
DEFAULT_ANTHROPIC_JUDGE = "claude-haiku-4-5"
EVAL_TEMPERATURE = 0.1
EVAL_SEED = 42
MAX_TOKENS = 1024
MAX_RETRIES = 3


@dataclass
class JudgeSpec:
    """One judge configuration.

    Attributes:
        provider: ``"openai"`` or ``"anthropic"``.
        model: Provider model id, e.g. ``"gpt-4o-mini-2024-07-18"`` or ``"claude-haiku-4-5"``.
        temperature: Sampling temperature (OpenAI only; the Claude call omits it because the
            newest Claude models reject it).
        seed: Sampling seed (OpenAI only); ``None`` omits the parameter.
        max_tokens: Response token cap.
        thinking: Anthropic only. ``None`` omits the parameter, which is correct for Haiku 4.5.
            Models that think adaptively by default bill thinking against ``max_tokens``; pass
            ``{"type": "disabled"}`` (where the model accepts it) to keep the JSON from being
            truncated.
        base_url: OpenAI only; any OpenAI-compatible endpoint.
    """

    provider: str
    model: str
    temperature: float = EVAL_TEMPERATURE
    seed: int | None = EVAL_SEED
    max_tokens: int = MAX_TOKENS
    thinking: dict[str, Any] | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown judge provider {self.provider!r}; expected one of {PROVIDERS}."
            )
        if not self.model:
            raise ValueError("JudgeSpec.model must be a non-empty model id.")

    @classmethod
    def parse(cls, text: str, **kwargs: Any) -> JudgeSpec:
        """Parse ``"<provider>:<model>"``, e.g. ``"openai:gpt-4o-mini-2024-07-18"``.

        Extra keyword arguments override the remaining fields (``temperature``, ``seed``, ...).
        """
        provider, sep, model = text.strip().partition(":")
        if not sep or not provider or not model:
            raise ValueError(
                f"Judge must look like '<provider>:<model>' "
                f"(e.g. 'openai:{DEFAULT_OPENAI_JUDGE}' or 'anthropic:{DEFAULT_ANTHROPIC_JUDGE}'); "
                f"got {text!r}."
            )
        return cls(provider=provider.strip().lower(), model=model.strip(), **kwargs)

    @property
    def tag(self) -> str:
        """Filesystem-safe name, e.g. ``anthropic_claude-haiku-4-5``."""
        return re.sub(r"[^A-Za-z0-9.\-]+", "_", f"{self.provider}_{self.model}")

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def make_judge_client(judge: JudgeSpec) -> Any:
    """Build the async SDK client for ``judge`` from environment credentials."""
    from .. import clients

    if judge.provider == "openai":
        return clients.make_openai_client(base_url=judge.base_url)
    return clients.make_anthropic_client()


async def call_openai_json(
    client: Any,
    prompt: str,
    schema: dict,
    schema_name: str = "evaluation",
    model: str = DEFAULT_OPENAI_JUDGE,
    temperature: float = EVAL_TEMPERATURE,
    max_retries: int = MAX_RETRIES,
    *,
    seed: int | None = EVAL_SEED,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """OpenAI chat completion with a strict JSON-schema response and exponential back-off.

    Raises:
        ValueError: when every attempt failed (the last error is included in the message).
    """
    extra: dict[str, Any] = {"seed": seed} if seed is not None else {}
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema, "strict": True},
                },
                **extra,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response")
            return json.loads(content)
        except Exception as e:
            if attempt >= max_retries - 1:
                raise ValueError(f"Failed after {max_retries} attempts: {e}") from e
            logger.debug("OpenAI judge attempt %d failed (%s); retrying", attempt + 1, e)
            await asyncio.sleep(2**attempt)
    raise ValueError("max_retries must be >= 1")


def _strip_unsupported_constraints(schema: dict) -> dict:
    """Remove ``minimum``/``maximum``/``minItems``/``maxItems``/``multipleOf`` recursively.

    Claude's JSON-schema structured output rejects numeric bounds and array-length constraints.
    Each dropped constraint is folded into the node's ``description`` instead of being lost: for
    the array-shaped questionnaires (Q1, Q2, WAI-SR, CSQ-8, MI-SAT) ``minItems == maxItems`` is
    the only signal that one score per item is required, and a wrong-length array would fail
    validation and turn into a missing score -- biased missingness on the headline metric.
    Counts and ranges are still re-validated client-side by ``parse_json_response``.
    """
    s = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            notes = []
            n_items = node.get("minItems")
            if (
                node.get("type") == "array"
                and n_items is not None
                and n_items == node.get("maxItems")
            ):
                notes.append(f"Return EXACTLY {int(n_items)} values, in item order.")
            lo, hi = node.get("minimum"), node.get("maximum")
            if lo is not None and hi is not None:
                notes.append(f"Integer from {lo} to {hi} inclusive.")
            elif lo is not None:
                notes.append(f"Integer >= {lo}.")
            if notes:
                node["description"] = " ".join(
                    [node.get("description", "").strip(), *notes]
                ).strip()
            for k in ("minimum", "maximum", "minItems", "maxItems", "multipleOf"):
                node.pop(k, None)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)
    return s


async def call_anthropic_json(
    client: Any,
    prompt: str,
    schema: dict,
    *,
    model: str,
    max_tokens: int = MAX_TOKENS,
    thinking: dict | None = None,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Claude structured output: the first text block of the response is valid JSON.

    No ``temperature`` is sent (rejected by the newest Claude models); ``thinking`` is passed
    through only when set.

    Raises:
        ValueError: when every attempt failed, including refusals and empty responses.
    """
    extra = {"thinking": thinking} if thinking is not None else {}
    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _strip_unsupported_constraints(schema),
                    }
                },
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            if response.stop_reason == "refusal":
                raise ValueError("Claude refused the request (stop_reason=refusal)")
            text = next((b.text for b in response.content if b.type == "text"), "")
            if not text.strip():
                raise ValueError("Empty response")
            return json.loads(text)
        except Exception as e:
            if attempt >= max_retries - 1:
                raise ValueError(f"Failed after {max_retries} attempts: {e}") from e
            logger.debug("Anthropic judge attempt %d failed (%s); retrying", attempt + 1, e)
            await asyncio.sleep(2**attempt)
    raise ValueError("max_retries must be >= 1")


async def call_judge_json(
    client: Any,
    judge: JudgeSpec,
    prompt: str,
    schema: dict,
    schema_name: str,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Dispatch one structured-JSON judge call on ``judge.provider``."""
    if judge.provider == "openai":
        return await call_openai_json(
            client,
            prompt,
            schema,
            schema_name=schema_name,
            model=judge.model,
            temperature=judge.temperature,
            max_retries=max_retries,
            seed=judge.seed,
            max_tokens=judge.max_tokens,
        )
    return await call_anthropic_json(
        client,
        prompt,
        schema,
        model=judge.model,
        max_tokens=judge.max_tokens,
        thinking=judge.thinking,
        max_retries=max_retries,
    )
