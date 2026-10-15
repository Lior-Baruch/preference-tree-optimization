"""Shared fixtures for the offline test suite.

Everything here runs on CPU without network access:

* ``FakeAsyncOpenAI`` stands in for both API roles. A request carrying ``response_format`` is
  answered as the oracle / judge (a schema-valid JSON object whose scores are a deterministic
  hash of the prompt, so different candidate transcripts get different scores and the tau
  filter can pass); any other request is answered as the simulated patient.
* ``FakeAsyncAnthropic`` mirrors the Messages API shape used by the held-out judge.
* ``tiny_tokenizer`` is a byte-level BPE trained in-process with ``tokenizers`` and loaded back
  through ``pto.model.setup_tokenizer``; ``make_tiny_model`` builds a 2-layer random Llama.
  Nothing is downloaded from the Hub.
"""

from __future__ import annotations

import os

# Must be set before anything imports torch: the suite is CPU-only by design.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import copy  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
CONFIGS_DIR = REPO_ROOT / "configs"
GOLDEN_PATH = TESTS_DIR / "fixtures" / "golden_prompts.json"

PATIENT_SENTENCES = (
    "I hear what you are saying, but it is hard for me.",
    "Honestly, I have tried before and it never lasts.",
    "My family keeps asking me to change, and I feel pressured.",
    "I guess I could think about cutting down a little.",
    "Work is stressful, and that is when it gets worse.",
    "I am not sure therapy is going to help me.",
    "That makes sense, I would like to feel healthier.",
    "Maybe I could try walking in the evenings.",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: end-to-end tests (tiny model + DPO on CPU); run by default"
    )


# ---------------------------------------------------------------------------------------------
# Deterministic JSON-schema instances (oracle / judge answers)
# ---------------------------------------------------------------------------------------------


def stable_hash(*parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _bounds_from_description(node: dict) -> tuple[int | None, int | None]:
    """Recover integer bounds that ``_strip_unsupported_constraints`` folded into text."""
    desc = node.get("description", "") or ""
    m = re.search(r"Integer from (-?\d+) to (-?\d+)", desc)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"Integer >= (-?\d+)", desc)
    if m:
        return int(m.group(1)), None
    return None, None


def _array_len(node: dict) -> int:
    if "minItems" in node:
        return int(node["minItems"])
    if "maxItems" in node:
        return int(node["maxItems"])
    m = re.search(r"EXACTLY (\d+) values", node.get("description", "") or "")
    return int(m.group(1)) if m else 3


IntChooser = Callable[[int, int | None, str], int]


def schema_instance(schema: dict, seed: str, int_fn: IntChooser | None = None, path: str = "$"):
    """Build a value that satisfies ``schema``.

    ``int_fn(lo, hi, path)`` picks bounded integers; the default is a hash of ``seed`` and the
    JSON path. Count fields (``minimum`` only) default to ``[lo, lo + 4]``.
    """
    if "enum" in schema:
        options = list(schema["enum"])
        if len(options) == 1 or schema.get("type") == "integer":
            return options[0]
        return options[stable_hash(seed, path) % len(options)]
    kind = schema.get("type")
    if kind == "object":
        return {
            key: schema_instance(sub, seed, int_fn, f"{path}.{key}")
            for key, sub in schema.get("properties", {}).items()
        }
    if kind == "array":
        items = schema.get("items", {})
        return [
            schema_instance(items, seed, int_fn, f"{path}[{i}]") for i in range(_array_len(schema))
        ]
    if kind == "integer":
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if lo is None and hi is None:
            lo, hi = _bounds_from_description(schema)
        lo = 0 if lo is None else int(lo)
        if int_fn is not None:
            return int(int_fn(lo, None if hi is None else int(hi), path))
        hi = lo + 4 if hi is None else int(hi)
        return lo + stable_hash(seed, path) % (hi - lo + 1)
    if kind == "number":
        return float(stable_hash(seed, path) % 5)
    if kind == "boolean":
        return False
    return "x"


def schema_qid(schema: dict) -> int:
    return int(schema["properties"]["questionnaire_id"]["enum"][0])


# ---------------------------------------------------------------------------------------------
# Fake API clients
# ---------------------------------------------------------------------------------------------


def _chat_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, refusal=None)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return SimpleNamespace(choices=[choice], usage=usage, model="fake")


OracleFn = Callable[[dict, str], Any]


class FakeAsyncOpenAI:
    """Async stand-in for ``openai.AsyncOpenAI`` covering ``chat.completions.create``.

    Args:
        end_after_patient_turns: the patient's N-th reply ends with ``SESSION ENDED`` (None =
            never). The turn number is read from the patient-perspective history (its own
            previous replies are the ``assistant`` messages).
        oracle_fn: ``(schema, prompt) -> dict | str`` overriding the oracle answer; a ``str``
            is returned verbatim as the message content.
        fail_first: the first N calls raise ``RuntimeError`` (transient-failure simulation).
    """

    def __init__(
        self,
        *,
        end_after_patient_turns: int | None = None,
        oracle_fn: OracleFn | None = None,
        fail_first: int = 0,
    ) -> None:
        self.end_after_patient_turns = end_after_patient_turns
        self.oracle_fn = oracle_fn
        self.fail_first = fail_first
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def oracle_calls(self) -> list[dict]:
        return [c for c in self.calls if "response_format" in c]

    @property
    def patient_calls(self) -> list[dict]:
        return [c for c in self.calls if "response_format" not in c]

    def oracle_qids(self) -> list[int]:
        return [
            schema_qid(c["response_format"]["json_schema"]["schema"]) for c in self.oracle_calls
        ]

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(copy.deepcopy(kwargs))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("simulated transient API failure")
        if "response_format" in kwargs:
            return _chat_response(self._oracle_content(kwargs))
        return _chat_response(self._patient_content(kwargs["messages"]))

    def _oracle_content(self, kwargs: dict) -> str:
        schema = kwargs["response_format"]["json_schema"]["schema"]
        prompt = kwargs["messages"][-1]["content"]
        if self.oracle_fn is not None:
            out = self.oracle_fn(schema, prompt)
            return out if isinstance(out, str) else json.dumps(out)
        return json.dumps(schema_instance(schema, prompt))

    def _patient_content(self, messages: list[dict]) -> str:
        turn = 1 + sum(1 for m in messages if m.get("role") == "assistant")
        h = stable_hash(json.dumps(messages, sort_keys=True))
        n = len(PATIENT_SENTENCES)
        text = f"{PATIENT_SENTENCES[h % n]} {PATIENT_SENTENCES[(h // n) % n]}"
        if self.end_after_patient_turns is not None and turn >= self.end_after_patient_turns:
            return f"{text} SESSION ENDED 2"
        return text


class FakeAsyncAnthropic:
    """Async stand-in for ``anthropic.AsyncAnthropic`` covering ``messages.create``.

    Reads the schema from ``output_config.format.schema`` (bounds may have been folded into
    ``description`` text by the constraint stripper) and answers with a valid instance.
    """

    def __init__(self, *, oracle_fn: OracleFn | None = None) -> None:
        self.oracle_fn = oracle_fn
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(copy.deepcopy(kwargs))
        fmt = (kwargs.get("output_config") or {}).get("format") or kwargs.get("output_format")
        schema = fmt["schema"]
        prompt = kwargs["messages"][-1]["content"]
        if self.oracle_fn is not None:
            out = self.oracle_fn(schema, prompt)
            text = out if isinstance(out, str) else json.dumps(out)
        else:
            text = json.dumps(schema_instance(schema, prompt))
        block = SimpleNamespace(type="text", text=text)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        return SimpleNamespace(content=[block], stop_reason="end_turn", usage=usage)


@pytest.fixture
def fake_openai() -> Callable[..., FakeAsyncOpenAI]:
    """Factory: ``fake_openai(end_after_patient_turns=..., oracle_fn=..., fail_first=...)``."""
    return FakeAsyncOpenAI


@pytest.fixture
def fake_anthropic() -> Callable[..., FakeAsyncAnthropic]:
    return FakeAsyncAnthropic


@pytest.fixture
def schema_tools() -> SimpleNamespace:
    """Expose the schema helpers to test modules without importing conftest."""
    return SimpleNamespace(
        instance=schema_instance, qid=schema_qid, stable_hash=stable_hash
    )


# ---------------------------------------------------------------------------------------------
# Config-object helpers (field sets are read from the dataclasses, so optional knobs keep
# their defaults)
# ---------------------------------------------------------------------------------------------


def _build_dataclass(cls: type, values: dict) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in names})


@pytest.fixture
def make_oracle_cfg() -> Callable[..., Any]:
    def _make(**overrides: Any) -> Any:
        from pto.oracle import OracleConfig

        values = {
            "model_id": "gpt-4o-mini-2024-07-18",
            "request_timeout": 5.0,
            "max_retries": 3,
            "eval_temperature": 0.0,
            "max_concurrency": 8,
            "min_success_ratio": 0.5,
        }
        values.update(overrides)
        return _build_dataclass(OracleConfig, values)

    return _make


@pytest.fixture
def make_lookahead_cfg() -> Callable[..., Any]:
    def _make(**overrides: Any) -> Any:
        from pto.lookahead import LookaheadConfig

        values = {
            "k": 0,
            "temperature_therapist": 0.9,
            "temperature_patient": 0.7,
            "max_tokens": 8,
            "max_input_tokens": 2048,
            "patient_model_id": "gpt-4o-mini-2024-07-18",
            "patient_api_concurrency": 8,
            "patient_api_max_retries": 1,
            "patient_api_backoff_seconds": 0.0,
            "stop_strings": ["<|im_end|>", "<|im_start|>"],
            "lookahead_sub_batch_size": None,
        }
        values.update(overrides)
        return _build_dataclass(LookaheadConfig, values)

    return _make


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``asyncio.sleep`` with a recorder so retry back-off costs no wall time."""
    import asyncio

    slept: list[float] = []

    async def _sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        slept.append(delay)
        return args[0] if args else None

    monkeypatch.setattr(asyncio, "sleep", _sleep)
    return slept


# ---------------------------------------------------------------------------------------------
# Golden instrument strings
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS_DIR


# ---------------------------------------------------------------------------------------------
# Tiny offline tokenizer + model
# ---------------------------------------------------------------------------------------------

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"


def _tokenizer_corpus() -> list[str]:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    corpus: list[str] = list(golden["patient_system_prompts"])
    corpus += [golden["therapist_system_prompt"], golden["therapist_init_utterance"]]
    corpus += [q["prompt"] for q in golden["questionnaires"].values()]
    corpus += list(PATIENT_SENTENCES)
    chatml = [
        f"<|im_start|>{role}\n{text}<|im_end|>\n"
        for role in ("system", "user", "assistant")
        for text in PATIENT_SENTENCES
    ]
    return corpus + chatml * 20


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A byte-level BPE trained in-process, saved as a HF tokenizer directory."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=2000,
        special_tokens=[BOS_TOKEN, EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_tokenizer_corpus(), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=EOS_TOKEN,
        model_max_length=4096,
    )
    out = tmp_path_factory.mktemp("tiny_tokenizer")
    fast.save_pretrained(str(out))
    return out


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_tokenizer_dir: Path) -> Any:
    """The tiny tokenizer configured exactly like the policy tokenizer (ChatML, left pad)."""
    from pto.model import setup_tokenizer

    return setup_tokenizer(str(tiny_tokenizer_dir))


def build_tiny_model(tokenizer: Any, seed: int = 0) -> Any:
    from pto import _compat  # noqa: F401  (trl before torch)

    # isort: split
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tokenizer),
        max_position_embeddings=4096,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.eval()
    return model


@pytest.fixture
def make_tiny_model(tiny_tokenizer: Any) -> Callable[..., Any]:
    """Factory for a fresh random 2-layer Llama (same seed -> identical base weights).

    Function-scoped on purpose: DPO/PEFT inject LoRA layers into the base modules in place,
    so a model must never be shared between tests.
    """

    def _make(seed: int = 0) -> Any:
        return build_tiny_model(tiny_tokenizer, seed=seed)

    return _make


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory, tiny_tokenizer: Any) -> Path:
    """The tiny model + tokenizer saved as a local HF checkpoint (for build-from-config paths)."""
    out = tmp_path_factory.mktemp("tiny_llama")
    model = build_tiny_model(tiny_tokenizer, seed=0)
    model.save_pretrained(str(out))
    tiny_tokenizer.save_pretrained(str(out))
    return out


# ---------------------------------------------------------------------------------------------
# Conversation-directory helper (layout written by the trainer / ``pto generate``)
# ---------------------------------------------------------------------------------------------


def write_conversation_dir(
    conv_dir: Path, conversations: list[list[str]], persona_ids: list[int]
) -> Path:
    """Write ``conversation_{i}.csv`` (therapist first, alternating) + ``manifest.json``."""
    import pandas as pd

    conv_dir.mkdir(parents=True, exist_ok=True)
    for i, utterances in enumerate(conversations):
        roles = ["therapist" if j % 2 == 0 else "patient" for j in range(len(utterances))]
        pd.DataFrame(
            {
                "role": roles,
                "conversation": utterances,
                "session_ended_by": None,
                "session_ended_explanation": None,
            }
        ).to_csv(conv_dir / f"conversation_{i}.csv", index=False)
    manifest = {str(i): int(p) for i, p in enumerate(persona_ids)}
    (conv_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return conv_dir


@pytest.fixture
def conversation_dir_writer() -> Callable[..., Path]:
    return write_conversation_dir
