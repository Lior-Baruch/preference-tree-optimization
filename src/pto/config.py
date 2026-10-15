"""Run configuration: nested dataclasses mirroring the YAML schema in ``configs/*.yaml``.

Defaults reproduce the reported runs (Llama-3.2-1B policy, gpt-4o-mini patient and oracle,
Q1+Q2 reward, greedy trees with M=8 branches, tau=0.1, MCL=12, K=5 look-ahead, 10 iterations).
See ``docs/configuration.md`` for every key.

This module only depends on PyYAML so that ``pto train --dry-run`` and config tests never import
torch.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "ConfigError",
    "DPOSettings",
    "GenerationSettings",
    "HubSettings",
    "LoggingSettings",
    "LookaheadSettings",
    "LoopSettings",
    "LoraSettings",
    "ModelSettings",
    "OracleSettings",
    "PTORunConfig",
    "TreeSettings",
    "dump_config",
    "load_config",
]

#: Questionnaires whose oracle answer is a flat ``scores`` array; only these can be a training
#: reward (the nested-schema instruments are evaluation-only).
FLAT_SCORE_QUESTIONNAIRES = (1, 2, 3, 4, 6)
#: Run-name token per questionnaire id; a multi-questionnaire oracle concatenates them
#: (``[1, 2]`` -> ``Q1Q2``).
_ORACLE_TOKENS = {1: "Q1", 2: "Q2", 3: "WAI", 4: "CSQ8", 6: "MI_SAT"}
DTYPES = ("auto", "bfloat16", "float16", "float32")
TREE_MODES = ("greedy", "independent")
_MODE_TOKENS = {"greedy": "greedy", "independent": "indep"}
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigError(ValueError):
    """Invalid or unknown configuration values."""


@dataclass
class ModelSettings:
    """Model identifiers and how the policy is loaded."""

    policy: str = "meta-llama/Llama-3.2-1B"
    tokenizer: str | None = None  # null -> same as policy
    patient: str = "gpt-4o-mini-2024-07-18"
    oracle: str = "gpt-4o-mini-2024-07-18"
    api_base_url: str | None = None  # any OpenAI-compatible endpoint; null -> api.openai.com
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32 (auto: bf16/fp16 on GPU, fp32 on CPU)
    load_in_4bit: bool = False


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj",
        ]
    )


@dataclass
class LoopSettings:
    num_iterations: int = 10
    conversations_per_iteration: int = 96
    conversation_length: int = 49  # utterances per generated conversation (therapist + patient)
    epochs_per_iteration: int = 2


@dataclass
class GenerationSettings:
    """Full-conversation generation (policy therapist vs. simulated patient)."""

    batch_size: int = 64
    therapist_temperature: float = 0.9
    patient_temperature: float = 0.7
    max_input_tokens: int = 2048
    max_new_tokens: int = 200
    patient_concurrency: int = 96
    patient_max_retries: int = 3
    patient_backoff_s: float = 1.0
    max_retries_without_progress: int = 3
    stop_strings: list[str] = field(default_factory=lambda: ["<|im_end|>", "<|im_start|>"])


@dataclass
class OracleSettings:
    questionnaires: list[int] = field(default_factory=lambda: [1, 2])
    temperature: float = 0.0
    max_retries: int = 3
    timeout_s: float = 60.0
    concurrency: int = 128


@dataclass
class LookaheadSettings:
    k: int = 5  # simulated turns after each candidate before scoring; 0 disables look-ahead
    therapist_temperature: float = 0.9
    patient_temperature: float = 0.7
    max_new_tokens: int = 200
    sub_batch_size: int | None = 128  # therapist generate() sub-batch; null = all at once


@dataclass
class TreeSettings:
    mode: str = "greedy"  # greedy (grow one trunk from the best branch) | independent
    branches: int = 8  # M candidate completions per branch point
    tau: float = 0.1  # emit a pair only when best - worst > tau (strict)
    temperature: float = 1.2
    max_new_tokens: int = 200
    min_conv_length: int = 12  # MCL: greedy seed length / independent branch-point floor
    trunk_target_len: int | None = 49  # greedy trunk cap (min with conversation_length)


@dataclass
class DPOSettings:
    learning_rate: float = 1.0e-5
    beta: float = 0.1
    loss_type: str = "sigmoid"
    precompute_ref_log_probs: bool = True
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.01
    eval_split: float = 0.05
    max_prompt_length: int = 2048
    max_completion_length: int = 200
    bf16: Any = "auto"  # auto (CUDA with bf16 support) | true | false


@dataclass
class LoggingSettings:
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    logging_steps: int = 1
    save_steps: int = 10
    save_total_limit: int | None = 2
    save_generations: bool = True  # iteration_N/eda/generations.jsonl (every candidate + score)
    save_lookahead_transcripts: bool = True  # keep the K-turn look-ahead text in those records


@dataclass
class HubSettings:
    push: bool = False
    repo_id: str | None = None


@dataclass
class PTORunConfig:
    """A complete PTO run. The YAML key ``run_name`` maps to :attr:`explicit_run_name`."""

    explicit_run_name: str | None = None
    output_dir: str = "runs"
    seed: int = 42
    models: ModelSettings = field(default_factory=ModelSettings)
    lora: LoraSettings = field(default_factory=LoraSettings)
    loop: LoopSettings = field(default_factory=LoopSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    oracle: OracleSettings = field(default_factory=OracleSettings)
    lookahead: LookaheadSettings = field(default_factory=LookaheadSettings)
    tree: TreeSettings = field(default_factory=TreeSettings)
    dpo: DPOSettings = field(default_factory=DPOSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    hub: HubSettings = field(default_factory=HubSettings)

    # ── Derived values ──────────────────────────────────────────────────────────

    @property
    def run_name(self) -> str:
        """Explicit ``run_name`` or ``pto_{oracle}_{policy}_K{k}_MCL{mcl}_M{m}_{mode}``."""
        if self.explicit_run_name:
            return self.explicit_run_name
        qids = sorted({int(q) for q in self.oracle.questionnaires})
        oracle = "".join(_ORACLE_TOKENS.get(q, f"Q{q}") for q in qids)
        mode = _MODE_TOKENS.get(self.tree.mode, self.tree.mode)
        return (
            f"pto_{oracle}_{_policy_short_name(self.models.policy)}_K{self.lookahead.k}"
            f"_MCL{self.tree.min_conv_length}_M{self.tree.branches}_{mode}"
        )

    @property
    def run_dir(self) -> Path:
        """``{output_dir}/{run_name}``: holds config, metadata, iterations and conversations."""
        return Path(self.output_dir) / self.run_name

    @property
    def tokenizer_id(self) -> str:
        return self.models.tokenizer or self.models.policy

    @property
    def effective_trunk_target_len(self) -> int:
        """Greedy trunk length cap: ``min(conversation_length, trunk_target_len)``."""
        if self.tree.trunk_target_len is None:
            return self.loop.conversation_length
        return min(self.loop.conversation_length, self.tree.trunk_target_len)

    # ── Validation / serialisation ──────────────────────────────────────────────

    def validate(self) -> PTORunConfig:
        """Raise :class:`ConfigError` listing every problem; returns ``self`` when valid."""
        errors: list[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        if self.explicit_run_name is not None:
            check(
                isinstance(self.explicit_run_name, str)
                and bool(_RUN_NAME_RE.match(self.explicit_run_name)),
                f"run_name ({self.explicit_run_name!r}) may only contain letters, digits, '.', "
                "'_' and '-'",
            )
        check(isinstance(self.seed, int), "seed must be an integer")

        m = self.models
        check(m.dtype in DTYPES, f"models.dtype ({m.dtype!r}) must be one of {list(DTYPES)}")
        check(bool(m.policy), "models.policy must be set")

        lp = self.loop
        check(lp.num_iterations > 0, "loop.num_iterations must be > 0")
        check(lp.epochs_per_iteration > 0, "loop.epochs_per_iteration must be > 0")
        check(lp.conversations_per_iteration > 0, "loop.conversations_per_iteration must be > 0")

        g = self.generation
        check(g.batch_size > 0, "generation.batch_size must be > 0")
        check(g.patient_concurrency > 0, "generation.patient_concurrency must be > 0")
        check(g.max_new_tokens > 0, "generation.max_new_tokens must be > 0")

        o = self.oracle
        qs = list(o.questionnaires)
        check(len(qs) > 0, "oracle.questionnaires must not be empty")
        check(
            all(isinstance(q, int) and q in FLAT_SCORE_QUESTIONNAIRES for q in qs),
            f"oracle.questionnaires ({qs}) must be a subset of {list(FLAT_SCORE_QUESTIONNAIRES)} "
            "(questionnaires with a flat score array)",
        )
        check(o.concurrency > 0, "oracle.concurrency must be > 0")
        check(o.timeout_s > 0, "oracle.timeout_s must be > 0")
        check(o.max_retries >= 1, "oracle.max_retries must be >= 1")

        la = self.lookahead
        check(isinstance(la.k, int) and la.k >= 0, f"lookahead.k ({la.k}) must be an integer >= 0")
        check(
            la.sub_batch_size is None or la.sub_batch_size > 0,
            "lookahead.sub_batch_size must be > 0 or null",
        )

        t = self.tree
        check(t.mode in TREE_MODES, f"tree.mode ({t.mode!r}) must be one of {list(TREE_MODES)}")
        check(t.branches >= 2, f"tree.branches ({t.branches}) must be >= 2 (a pair needs two)")
        check(t.tau >= 0, f"tree.tau ({t.tau}) must be >= 0")
        check(t.min_conv_length >= 2, f"tree.min_conv_length ({t.min_conv_length}) must be >= 2")
        if t.mode == "greedy":
            check(
                t.min_conv_length % 2 == 0,
                f"tree.mode 'greedy' needs an EVEN tree.min_conv_length (got {t.min_conv_length}) "
                "so the seed prefix ends on a patient turn",
            )
            if t.trunk_target_len is not None:
                check(
                    t.trunk_target_len >= t.min_conv_length + 1,
                    f"tree.trunk_target_len ({t.trunk_target_len}) must be >= "
                    f"tree.min_conv_length + 1 ({t.min_conv_length + 1})",
                )
        check(
            lp.conversation_length >= t.min_conv_length + 1,
            f"loop.conversation_length ({lp.conversation_length}) must be >= "
            f"tree.min_conv_length + 1 ({t.min_conv_length + 1}); shorter conversations never "
            "reach a branch point",
        )

        d = self.dpo
        check(0.0 < d.eval_split < 1.0, f"dpo.eval_split ({d.eval_split}) must be in (0, 1)")
        check(
            d.max_prompt_length > 0 and d.max_completion_length > 0,
            "dpo.max_prompt_length and dpo.max_completion_length must be > 0",
        )
        check(
            d.bf16 == "auto" or isinstance(d.bf16, bool),
            f"dpo.bf16 ({d.bf16!r}) must be 'auto', true or false",
        )
        check(d.per_device_train_batch_size > 0, "dpo.per_device_train_batch_size must be > 0")
        check(d.gradient_accumulation_steps > 0, "dpo.gradient_accumulation_steps must be > 0")
        check(0.0 <= d.warmup_ratio < 1.0, "dpo.warmup_ratio must be in [0, 1)")

        lg = self.logging
        check(
            isinstance(lg.report_to, list) and all(isinstance(r, str) for r in lg.report_to),
            "logging.report_to must be a list of strings",
        )
        check(lg.save_steps > 0, "logging.save_steps must be > 0")
        check(
            lg.save_total_limit is None or lg.save_total_limit > 0,
            "logging.save_total_limit must be > 0 or null",
        )

        if self.hub.push:
            check(bool(self.hub.repo_id), "hub.repo_id is required when hub.push is true")

        if errors:
            raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in YAML layout, with the resolved ``run_name``."""
        data = dataclasses.asdict(self)
        data.pop("explicit_run_name")
        return {"run_name": self.run_name, **data}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> PTORunConfig:
        """Build from a YAML-layout mapping; unknown keys raise :class:`ConfigError`."""
        data = dict(data or {})
        if "run_name" in data:
            data["explicit_run_name"] = data.pop("run_name")
        elif "explicit_run_name" in data:
            raise ConfigError("Unknown key 'explicit_run_name' (use 'run_name')")
        return _build(cls, data, "")


# ── Loading / dumping ───────────────────────────────────────────────────────────


def _policy_short_name(model_id: str) -> str:
    """``meta-llama/Llama-3.2-1B`` -> ``Llama32-1B`` (basename, no dots, no letter-digit dash)."""
    name = model_id.rstrip("/").replace("\\", "/").split("/")[-1].replace(".", "")
    name = re.sub(r"(?<=[A-Za-z])-(?=\d)", "", name)
    return re.sub(r"[^A-Za-z0-9_-]", "-", name) or "policy"


def _coerce(value: Any, default: Any, key: str) -> Any:
    """Light type coercion against the field default (YAML reads ``1e-5`` as a string)."""
    if value is None or default is None:
        return value
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{key}: expected true/false, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool):
            raise ConfigError(f"{key}: expected a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{key}: expected a number, got {value!r}") from None
    if isinstance(default, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
        return value
    if isinstance(default, list):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{key}: expected a list, got {value!r}")
        return list(value)
    return value


def _field_default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


def _build(cls: type, data: Any, prefix: str) -> Any:
    where = prefix.rstrip(".") or "config"
    if data is None:
        return cls()
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(str(k) for k in data if k not in known)
    if unknown:
        allowed = sorted("run_name" if k == "explicit_run_name" else k for k in known)
        raise ConfigError(f"Unknown key(s) in {where}: {unknown}. Allowed: {allowed}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        default = _field_default(known[name])
        key = f"{prefix}{name}"
        if dataclasses.is_dataclass(default):
            kwargs[name] = _build(type(default), value, f"{key}.")
        else:
            kwargs[name] = _coerce(value, default, key)
    return cls(**kwargs)


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Merge ``overrides`` into ``raw``: dotted keys (``loop.num_iterations``) or nested dicts."""
    for key, value in overrides.items():
        parts = str(key).split(".")
        node = raw
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        leaf = parts[-1]
        if isinstance(value, Mapping) and isinstance(node.get(leaf), dict):
            _apply_overrides(node[leaf], value)
        elif isinstance(value, Mapping):
            node[leaf] = {}
            _apply_overrides(node[leaf], value)
        else:
            node[leaf] = value


def load_config(
    path: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
) -> PTORunConfig:
    """Load a YAML config (``None`` = all defaults), apply ``overrides`` and validate.

    ``overrides`` accepts dotted keys (``{"loop.num_iterations": 2}``) or nested mappings.
    Unknown keys anywhere raise :class:`ConfigError`.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        raw = loaded or {}
    if overrides:
        _apply_overrides(raw, overrides)
    return PTORunConfig.from_dict(raw).validate()


def dump_config(cfg: PTORunConfig, path: str | Path) -> Path:
    """Write ``cfg`` as YAML (with the resolved ``run_name``) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, default_flow_style=False)
    return path


def config_to_yaml(cfg: PTORunConfig) -> str:
    """The resolved config as a YAML string (used by ``pto train --dry-run``)."""
    return yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False)
