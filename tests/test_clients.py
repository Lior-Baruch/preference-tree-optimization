"""Environment-driven client construction (no request is ever sent)."""

from __future__ import annotations

import importlib.util

import pytest

from pto.clients import hf_login, make_anthropic_client, make_openai_client


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def test_openai_client_requires_key() -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_openai_client()


def test_openai_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    client = make_openai_client()
    assert client.api_key == "sk-test-not-a-real-key"
    assert hasattr(client.chat.completions, "create")


def test_openai_compatible_endpoint_without_key() -> None:
    client = make_openai_client(base_url="http://localhost:8000/v1")
    assert str(client.base_url).startswith("http://localhost:8000/v1")


def test_anthropic_client() -> None:
    if importlib.util.find_spec("anthropic") is None:
        with pytest.raises(ImportError, match="eval"):
            make_anthropic_client()
        return
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_anthropic_client()
    assert make_anthropic_client(api_key="sk-ant-test").api_key == "sk-ant-test"


def test_hf_login_is_noop_without_token() -> None:
    assert hf_login() is False
