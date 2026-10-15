"""Trainer pre-flight checks (before any model or API work), DPO arguments and checkpoint resume."""

from __future__ import annotations

import importlib.util
import math

import pytest

from pto.config import load_config


def _hide_modules(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name in names:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)


def test_missing_reporting_backend_fails_before_any_work(monkeypatch, tmp_path):
    from pto import trainer

    cfg = load_config(None, {"output_dir": str(tmp_path), "logging.report_to": ["wandb"]})
    _hide_modules(monkeypatch, "wandb")

    def no_client(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("client built before the reporting check")

    monkeypatch.setattr(trainer, "make_openai_client", no_client)
    with pytest.raises(ImportError, match="tracking"):
        trainer.train(cfg)
    assert not cfg.run_dir.exists()


def test_reporting_backend_check_accepts_alternatives(monkeypatch):
    from pto.trainer import _check_reporting_backends

    _check_reporting_backends([])
    _check_reporting_backends(["none-of-the-known-backends"])
    # tensorboardX is an accepted substitute for tensorboard.
    _hide_modules(monkeypatch, "tensorboard")
    if importlib.util.find_spec("tensorboardX") is None:
        with pytest.raises(ImportError, match="tensorboard"):
            _check_reporting_backends(["tensorboard"])
    else:  # pragma: no cover - depends on the environment
        _check_reporting_backends(["tensorboard"])


def test_build_dpo_args_passes_config_through(tmp_path):
    from pto import trainer

    # Non-default values, so a kwarg that silently fell back to TRL's default would show.
    cfg = load_config(None, {
        "output_dir": str(tmp_path), "dpo.beta": 0.3, "dpo.loss_type": "ipo", "seed": 7,
        "dpo.warmup_ratio": 0.1, "logging.report_to": [],
    })
    d = cfg.dpo
    args = trainer._build_dpo_args(cfg, str(tmp_path / "inner"), num_train_pairs=100)
    assert args.beta == 0.3
    assert args.loss_type in ("ipo", ["ipo"])
    assert args.seed == 7
    assert args.max_length == d.max_prompt_length + d.max_completion_length
    assert args.lr_scheduler_type == "cosine"
    steps_per_epoch = math.ceil(100 / (d.per_device_train_batch_size
                                       * d.gradient_accumulation_steps))
    assert args.warmup_steps == math.ceil(steps_per_epoch * cfg.loop.epochs_per_iteration * 0.1)
    assert args.num_train_epochs == cfg.loop.epochs_per_iteration
    assert args.precompute_ref_log_probs is True
    assert args.eval_strategy == "epoch"

    no_eval = trainer._build_dpo_args(cfg, str(tmp_path / "inner"), 100, has_eval=False)
    assert no_eval.eval_strategy == "no"


def test_resume_walks_back_to_last_complete_checkpoint(tmp_path):
    from pto.model import (
        HF_TRAINER_FILES,
        compute_cumulative_step_offset,
        get_latest_valid_hf_checkpoint,
    )

    training_dir = tmp_path / "iteration_1" / "training"
    complete, partial = training_dir / "checkpoint-2", training_dir / "checkpoint-4"
    for ckpt, files in ((complete, HF_TRAINER_FILES), (partial, HF_TRAINER_FILES[:-1])):
        ckpt.mkdir(parents=True)
        for name in files:
            (ckpt / name).write_text("{}", encoding="utf-8")

    # checkpoint-4 lacks trainer_state.json (a crash mid-write), so resume uses checkpoint-2.
    assert get_latest_valid_hf_checkpoint(training_dir) == str(complete)
    assert compute_cumulative_step_offset(tmp_path) == 2
