"""YAML config loading, defaults, validation, overrides and run naming."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from pto.config import ConfigError, PTORunConfig, dump_config, load_config


def _write(tmp_path: Path, data: dict, name: str = "cfg.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults_are_the_reported_runs() -> None:
    cfg = load_config(None)
    assert cfg.seed == 42
    assert cfg.models.policy == "meta-llama/Llama-3.2-1B"
    assert cfg.models.patient == cfg.models.oracle == "gpt-4o-mini-2024-07-18"
    assert cfg.models.dtype == "auto" and cfg.models.load_in_4bit is False
    assert (cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout) == (16, 16, 0.05)
    assert cfg.loop.num_iterations == 10
    assert cfg.loop.conversations_per_iteration == 96
    assert cfg.loop.conversation_length == 49
    assert cfg.loop.epochs_per_iteration == 2
    assert cfg.generation.stop_strings == ["<|im_end|>", "<|im_start|>"]
    assert (cfg.generation.therapist_temperature, cfg.generation.patient_temperature) == (0.9, 0.7)
    assert cfg.oracle.questionnaires == [1, 2]
    assert cfg.oracle.temperature == 0.0
    assert cfg.lookahead.k == 5
    assert (cfg.tree.mode, cfg.tree.branches, cfg.tree.tau) == ("greedy", 8, 0.1)
    assert (cfg.tree.temperature, cfg.tree.min_conv_length) == (1.2, 12)
    assert (cfg.dpo.learning_rate, cfg.dpo.beta, cfg.dpo.loss_type) == (1e-5, 0.1, "sigmoid")
    assert cfg.dpo.precompute_ref_log_probs is True
    assert cfg.dpo.bf16 == "auto"
    assert cfg.logging.report_to == ["tensorboard"]
    assert cfg.hub.push is False and cfg.hub.repo_id is None


@pytest.mark.parametrize(
    ("name", "k"), [("pto_k5.yaml", 5), ("pto_k0.yaml", 0), ("quicktest.yaml", None)]
)
def test_shipped_configs_load(configs_dir: Path, name: str, k: int | None) -> None:
    cfg = load_config(configs_dir / name)
    if k is not None:
        assert cfg.lookahead.k == k
        # The reported configs differ from the defaults only in K.
        expected = dataclasses.replace(
            PTORunConfig(), lookahead=dataclasses.replace(PTORunConfig().lookahead, k=k)
        )
        assert cfg == expected
    else:
        assert cfg.loop.num_iterations <= 2
        assert cfg.loop.conversations_per_iteration <= 8
        assert cfg.lookahead.k <= 2


def test_run_name_derivation(configs_dir: Path) -> None:
    assert load_config(configs_dir / "pto_k5.yaml").run_name == (
        "pto_Q1Q2_Llama32-1B_K5_MCL12_M8_greedy"
    )
    assert load_config(configs_dir / "pto_k0.yaml").run_name == (
        "pto_Q1Q2_Llama32-1B_K0_MCL12_M8_greedy"
    )
    cfg = load_config(None, {"tree.mode": "independent", "tree.min_conv_length": 11,
                             "tree.branches": 4, "oracle.questionnaires": [2]})
    assert cfg.run_name == "pto_Q2_Llama32-1B_K5_MCL11_M4_indep"


def test_explicit_run_name_and_run_dir(tmp_path: Path) -> None:
    cfg = load_config(None, {"run_name": "my-run", "output_dir": str(tmp_path)})
    assert cfg.run_name == "my-run"
    assert cfg.run_dir == tmp_path / "my-run"
    derived = load_config(None, {"output_dir": str(tmp_path)})
    assert derived.run_dir == tmp_path / derived.run_name


def test_overrides_dotted_and_nested(tmp_path: Path) -> None:
    path = _write(tmp_path, {"loop": {"num_iterations": 4}, "tree": {"branches": 6}})
    cfg = load_config(path, {"loop.num_iterations": 2, "tree": {"tau": 0.3}})
    assert cfg.loop.num_iterations == 2
    assert cfg.tree.branches == 6  # untouched file value survives a nested override
    assert cfg.tree.tau == 0.3


def test_yaml_scientific_notation_is_a_float(tmp_path: Path) -> None:
    # PyYAML reads "1e-5" (no dot) as a string; the loader must coerce it.
    path = tmp_path / "lr.yaml"
    path.write_text("dpo:\n  learning_rate: 1e-5\n", encoding="utf-8")
    assert load_config(path).dpo.learning_rate == pytest.approx(1e-5)


@pytest.mark.parametrize(
    "data",
    [
        {"bogus": 1},
        {"loop": {"num_iterations": 2, "typo_key": 3}},
        {"tree": {"M": 8}},
        {"explicit_run_name": "x"},
    ],
)
def test_unknown_keys_rejected(tmp_path: Path, data: dict) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize(
    "overrides",
    [
        {"tree.min_conv_length": 11},  # greedy needs an even MCL
        {"oracle.questionnaires": [7]},  # MITI has no flat score array
        {"oracle.questionnaires": [1, 10]},
        {"oracle.questionnaires": []},
        {"tree.branches": 1},
        {"lookahead.k": -1},
        {"tree.mode": "beam"},
        {"tree.tau": -0.1},
        {"models.dtype": "int8"},
        {"dpo.eval_split": 0.0},
        {"dpo.bf16": "yes"},
        {"loop.conversation_length": 12},  # never reaches a branch point with MCL 12
        {"hub.push": True},  # needs hub.repo_id
        {"run_name": "bad name/with slash"},
        {"loop.num_iterations": "two"},
    ],
)
def test_validation_errors(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        load_config(None, overrides)


def test_odd_mcl_allowed_in_independent_mode() -> None:
    cfg = load_config(None, {"tree.mode": "independent", "tree.min_conv_length": 11})
    assert cfg.tree.min_conv_length == 11


def test_validate_returns_self() -> None:
    cfg = PTORunConfig()
    assert cfg.validate() is cfg
    cfg.tree.branches = 1
    with pytest.raises(ConfigError):
        cfg.validate()


def test_dump_and_reload_round_trip(tmp_path: Path) -> None:
    cfg = load_config(None, {"loop.num_iterations": 3, "lookahead.k": 0, "tree.tau": 0.25})
    path = dump_config(cfg, tmp_path / "resolved" / "config.yaml")
    assert path.exists()
    reloaded = load_config(path)
    # The dump carries the resolved run name, so it reloads as an explicit one.
    assert reloaded.run_name == cfg.run_name
    assert dataclasses.replace(reloaded, explicit_run_name=None) == cfg
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["run_name"] == cfg.run_name
    assert list(raw)[:3] == ["run_name", "output_dir", "seed"]
