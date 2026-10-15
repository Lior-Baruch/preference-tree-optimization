"""API clients and Hub login, configured from environment variables.

- ``OPENAI_API_KEY``: patient simulator and training oracle (any OpenAI-compatible endpoint).
- ``ANTHROPIC_API_KEY``: optional held-out evaluation judge.
- ``HF_TOKEN``: gated base models (e.g. Llama) and optional adapter push to the Hub.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["MissingAPIKeyError", "hf_login", "make_anthropic_client", "make_openai_client"]

# The OpenAI SDK refuses to build a client without a key, but self-hosted OpenAI-compatible
# servers (vLLM, llama.cpp, ...) usually ignore it.
_PLACEHOLDER_KEY = "EMPTY"


class MissingAPIKeyError(RuntimeError):
    """A required API key environment variable is not set."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def make_openai_client(api_key: str | None = None, base_url: str | None = None) -> Any:
    """Return an ``openai.AsyncOpenAI`` client.

    Args:
        api_key: Explicit key; defaults to ``OPENAI_API_KEY``.
        base_url: Optional OpenAI-compatible endpoint (``models.api_base_url`` in the config).
            With a custom endpoint and no key, a placeholder key is used.

    Raises:
        MissingAPIKeyError: If no key is available for the default OpenAI endpoint
            (a ``RuntimeError`` subclass).
    """
    from openai import AsyncOpenAI

    key = (api_key or "").strip() or _env("OPENAI_API_KEY")
    if not key:
        if base_url is None:
            raise MissingAPIKeyError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable "
                "(or pass api_key=...)."
            )
        key = _PLACEHOLDER_KEY
        logger.info("No OPENAI_API_KEY set; using a placeholder key for %s", base_url)

    client = AsyncOpenAI(api_key=key, base_url=base_url)
    logger.info("OpenAI client initialized (%s)", base_url or "api.openai.com")
    return client


def make_anthropic_client(api_key: str | None = None) -> Any:
    """Return an ``anthropic.AsyncAnthropic`` client (held-out evaluation judge).

    Args:
        api_key: Explicit key; defaults to ``ANTHROPIC_API_KEY``.

    Raises:
        ImportError: If the ``anthropic`` package (the ``eval`` extra) is not installed.
        MissingAPIKeyError: If no key is available (a ``RuntimeError`` subclass).
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError as e:
        raise ImportError(
            "The Anthropic judge needs the 'anthropic' package. Install the 'eval' extra, "
            "e.g. pip install -e \".[eval]\" (or pip install anthropic)."
        ) from e

    key = (api_key or "").strip() or _env("ANTHROPIC_API_KEY")
    if not key:
        raise MissingAPIKeyError(
            "Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable "
            "(or pass api_key=...)."
        )
    return AsyncAnthropic(api_key=key)


def hf_login(token: str | None = None) -> bool:
    """Log in to the Hugging Face Hub with ``token`` or ``HF_TOKEN``.

    Without a token this is a no-op (public models still load; gated ones such as Llama and
    Hub pushes will fail later with an authentication error).

    Returns:
        ``True`` if a login was performed.
    """
    tok = (token or "").strip() or _env("HF_TOKEN") or _env("HUGGINGFACE_TOKEN")
    if not tok:
        logger.info("No HF_TOKEN set; skipping Hugging Face login")
        return False

    from huggingface_hub import login

    login(token=tok)
    logger.info("Logged in to the Hugging Face Hub")
    return True
