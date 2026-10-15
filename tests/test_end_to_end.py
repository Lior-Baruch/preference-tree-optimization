"""End-to-end PTO on CPU: tiny Llama policy, fake patient/oracle, real DPO.

Covers the full loop (generate -> greedy or independent tree -> DPO -> save), resume of finished
iterations and of a crash between the tree build and the adapter save, several iterations in one
process, conversation generation from a saved adapter, and judge scoring of the result.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pandas as pd
import pytest

from pto.config import load_config
from pto.evaluation import JudgeSpec, evaluate_run

pytestmark = pytest.mark.slow

N_CONVS = 4
LENGTH = 16
MCL = 12
TAU = 0.1


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Never log in to the Hub or reach an API from the end-to-end runs.
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY", "WANDB_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _config(configs_dir: Path, output_dir: Path, **extra):
    overrides = {
        "run_name": None,  # derived name, so run settings decide the run directory
        "output_dir": str(output_dir),
        "loop.num_iterations": 1,
        "loop.conversations_per_iteration": N_CONVS,
        "loop.conversation_length": LENGTH,
        "loop.epochs_per_iteration": 1,
        "generation.batch_size": N_CONVS,
        "generation.max_new_tokens": 12,
        "generation.max_input_tokens": 2048,
        "generation.patient_concurrency": 8,
        "generation.patient_backoff_s": 0.0,
        "oracle.concurrency": 16,
        "lookahead.k": 1,
        "lookahead.max_new_tokens": 12,
        "lookahead.sub_batch_size": None,
        "tree.mode": "greedy",
        "tree.branches": 3,
        "tree.tau": TAU,
        "tree.max_new_tokens": 12,
        "tree.min_conv_length": MCL,
        "tree.trunk_target_len": LENGTH,
        "dpo.learning_rate": 1.0e-3,
        "dpo.warmup_ratio": 0.0,
        "dpo.per_device_train_batch_size": 2,
        "dpo.per_device_eval_batch_size": 2,
        "dpo.gradient_accumulation_steps": 1,
        "dpo.gradient_checkpointing": False,
        "dpo.max_prompt_length": 2048,
        "dpo.max_completion_length": 16,
        "dpo.bf16": "auto",  # resolves to False on CPU
        "logging.report_to": [],
        "logging.save_steps": 50,
        "logging.save_generations": True,
        "hub.push": False,
    }
    overrides.update(extra)
    return load_config(configs_dir / "quicktest.yaml", overrides)


def _persona_order(seed: int, n: int) -> list[int]:
    """Persona ids of a conversation set, replaying the original persona shuffle."""
    order = list(range(96))
    random.Random(seed).shuffle(order)
    return order[:n]


def _check_conversations(conv_dir: Path, seed: int) -> None:
    files = sorted(conv_dir.glob("conversation_*.csv"))
    assert len(files) == N_CONVS
    manifest = json.loads((conv_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {str(i): pid for i, pid in enumerate(_persona_order(seed, N_CONVS))}
    for f in files:
        df = pd.read_csv(f, keep_default_na=False)
        assert list(df.columns) == [
            "role", "conversation", "session_ended_by", "session_ended_explanation",
        ]
        assert 2 <= len(df) <= LENGTH + 1
        assert df["role"].iloc[0] == "therapist"


def _check_pairs(pref_dir: Path, tokenizer, max_prompt_tokens: int) -> pd.DataFrame:
    pairs = pd.read_csv(pref_dir / "pairs.csv", keep_default_na=False)
    assert len(pairs) > 0
    assert (pairs["chosen_score"] - pairs["rejected_score"] > TAU).all()
    assert (pairs["chosen"] != pairs["rejected"]).all()
    for prompt in pairs["prompt"]:
        assert prompt.endswith("<|im_start|>assistant\n")
        assert len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_prompt_tokens
    assert (pref_dir / "pairs.meta.json").exists()
    assert not (pref_dir / "_progress.json").exists()
    return pairs


def _check_adapter(iter_dir: Path) -> None:
    from safetensors.torch import load_file

    adapter = iter_dir / "adapter"
    assert (adapter / "adapter_config.json").exists()
    weights = load_file(str(adapter / "adapter_model.safetensors"))
    lora_b = [t for k, t in weights.items() if "lora_B" in k]
    assert lora_b, "no LoRA weights saved"
    # LoRA B starts at zero; a non-zero B proves DPO actually updated the adapter.
    assert any(float(t.abs().sum()) > 0 for t in lora_b)
    # Only the policy adapter is saved, not TRL's frozen reference copy.
    assert not (adapter / "ref").exists()
    meta = json.loads((iter_dir / "iteration_metadata.json").read_text(encoding="utf-8"))
    assert meta["num_pref_pairs"] > 0


def _adapter_weights(iter_dir: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(iter_dir / "adapter" / "adapter_model.safetensors"))


def test_greedy_train_resume_generate_evaluate(
    tmp_path, configs_dir, fake_openai, tiny_tokenizer, make_tiny_model, monkeypatch
) -> None:
    import torch
    from peft import get_peft_model_state_dict

    import pto.trainer as trainer_mod
    from pto.trainer import generate, train

    cfg1 = _config(configs_dir, tmp_path / "runs")
    assert cfg1.dpo.precompute_ref_log_probs is True
    assert cfg1.run_name.endswith("_K1_MCL12_M3_greedy")

    # ── Run 1: one iteration ──
    client1 = fake_openai()
    run_dir = train(cfg1, client=client1, tokenizer=tiny_tokenizer, model=make_tiny_model())
    assert Path(run_dir) == cfg1.run_dir
    assert (run_dir / "config.yaml").exists() and (run_dir / "run_metadata.json").exists()
    assert client1.oracle_calls and client1.patient_calls
    # Look-ahead K=1: every scored candidate got one simulated patient turn first.
    assert len(client1.patient_calls) > N_CONVS * LENGTH // 2

    convs = run_dir / "conversations"
    _check_conversations(convs / "model_iter_0", seed=cfg1.seed + 1)
    _check_conversations(convs / "model_iter_1", seed=cfg1.seed + 1 + 1)  # final set
    it1 = run_dir / "iteration_1"
    pairs1 = _check_pairs(it1 / "pref_pairs", tiny_tokenizer, cfg1.dpo.max_prompt_length)
    assert set(pairs1["branch_depth"]) <= {12, 14}
    _check_adapter(it1)
    generations = it1 / "eda" / "generations.jsonl"
    assert generations.exists()
    # The recorded look-ahead proves K=1 was applied: one simulated patient turn per scored
    # candidate (unscored candidates record no look-ahead).
    recs = [
        json.loads(line)
        for line in generations.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    las = [c["lookahead"] for r in recs for c in r["candidates"] if c["lookahead"] is not None]
    assert las and all(la["k"] == 1 for la in las)
    assert any(
        la["realized_turns"] == 1 and la["tail"].startswith("\n\n[PATIENT]: ") for la in las
    )

    adapter_file = it1 / "adapter" / "adapter_model.safetensors"
    adapter_mtime = adapter_file.stat().st_mtime_ns
    pairs_bytes = (it1 / "pref_pairs" / "pairs.csv").read_bytes()
    final_set = {p.name: p.read_bytes() for p in (convs / "model_iter_1").glob("*.csv")}

    # ── Run 2: same run, two iterations -> iteration 1 is skipped, iteration 2 runs ──
    # Spy on the policy iteration 2 starts from: it must be iteration 1's trained adapter.
    seen: dict = {}
    real_run_one_iteration = trainer_mod.run_one_iteration

    def spy(**kw):
        seen.setdefault("sd", {
            k: v.detach().clone() for k, v in get_peft_model_state_dict(kw["policy"]).items()
        })
        return real_run_one_iteration(**kw)

    monkeypatch.setattr(trainer_mod, "run_one_iteration", spy)
    cfg2 = _config(configs_dir, tmp_path / "runs", **{"loop.num_iterations": 2})
    assert cfg2.run_dir == cfg1.run_dir
    client2 = fake_openai()
    run_dir2 = train(cfg2, client=client2, tokenizer=tiny_tokenizer, model=make_tiny_model())
    monkeypatch.setattr(trainer_mod, "run_one_iteration", real_run_one_iteration)
    assert run_dir2 == run_dir
    assert adapter_file.stat().st_mtime_ns == adapter_mtime
    saved = _adapter_weights(it1)
    assert seen["sd"].keys() == saved.keys()
    assert all(torch.equal(seen["sd"][k], saved[k]) for k in saved)
    assert (it1 / "pref_pairs" / "pairs.csv").read_bytes() == pairs_bytes
    # model_iter_1 (iteration 2's Step 1) was resumed from disk, not regenerated.
    assert {p.name: p.read_bytes() for p in (convs / "model_iter_1").glob("*.csv")} == final_set
    it2 = run_dir / "iteration_2"
    _check_pairs(it2 / "pref_pairs", tiny_tokenizer, cfg2.dpo.max_prompt_length)
    _check_adapter(it2)
    _check_conversations(convs / "model_iter_2", seed=cfg2.seed + 2 + 1)

    # ── Run 3: everything done -> no API traffic at all ──
    client3 = fake_openai()
    train(cfg2, client=client3, tokenizer=tiny_tokenizer, model=make_tiny_model())
    assert client3.calls == []
    assert not (run_dir / "iteration_3").exists()

    # ── Run 4: crash after pairs.csv but before the adapter save -> pairs reused, DPO reruns ──
    shutil.rmtree(it2 / "adapter")
    client4 = fake_openai()
    train(cfg2, client=client4, tokenizer=tiny_tokenizer, model=make_tiny_model())
    assert client4.oracle_calls == [] and client4.patient_calls == []
    assert (it2 / "adapter").is_dir()

    # ── Run 5: same, but tau changed -> the stale pairs.csv is rebuilt, not reused ──
    shutil.rmtree(it2 / "adapter")
    cfg5 = _config(configs_dir, tmp_path / "runs", **{"loop.num_iterations": 2, "tree.tau": 0.2})
    assert cfg5.run_dir == cfg1.run_dir  # tau is not part of the derived run name
    client5 = fake_openai()
    train(cfg5, client=client5, tokenizer=tiny_tokenizer, model=make_tiny_model())
    assert client5.oracle_calls
    meta = json.loads((it2 / "pref_pairs" / "pairs.meta.json").read_text(encoding="utf-8"))
    assert meta["config_key"]["pref_filter_tau"] == 0.2
    assert (it2 / "adapter").is_dir()

    # ── Generate with the trained adapter, then score with a (fake) judge ──
    out_dir = tmp_path / "generated"
    gen_client = fake_openai()
    generate(
        cfg2, str(it2 / "adapter"), out_dir, num_conversations=2, seed=7,
        client=gen_client, tokenizer=tiny_tokenizer, model=make_tiny_model(),
    )
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {str(i): pid for i, pid in enumerate(_persona_order(7, 2))}
    assert sorted(p.name for p in out_dir.glob("conversation_*.csv")) == [
        "conversation_0.csv", "conversation_1.csv",
    ]
    assert gen_client.patient_calls and not gen_client.oracle_calls

    judge_client = fake_openai()
    scores = evaluate_run(
        out_dir, JudgeSpec.parse("openai:gpt-4o-mini-2024-07-18"),
        instruments=["Q1", "Q2", "PCT"], out_csv=tmp_path / "scores.csv", client=judge_client,
    )
    assert set(scores["instrument"]) <= {"Q1", "Q2", "Q1+Q2", "PCT"}
    assert {"Q1", "Q2", "Q1+Q2"} <= set(scores["instrument"])
    assert set(scores["persona_id"]) == set(manifest.values())


def test_three_iterations_in_one_process(
    tmp_path, configs_dir, fake_openai, tiny_tokenizer, make_tiny_model
) -> None:
    """From iteration 2 on, DPO receives the previous PeftModel; TRL's frozen "ref" adapter of
    the previous iteration must not linger (newer peft raises on a duplicate adapter name)."""
    import torch

    from pto.trainer import train

    cfg = _config(configs_dir, tmp_path / "runs", **{"loop.num_iterations": 3})
    run_dir = train(cfg, client=fake_openai(), tokenizer=tiny_tokenizer, model=make_tiny_model())
    for n in (1, 2, 3):
        _check_adapter(run_dir / f"iteration_{n}")
    w2, w3 = _adapter_weights(run_dir / "iteration_2"), _adapter_weights(run_dir / "iteration_3")
    assert w2.keys() == w3.keys()
    assert any(not torch.equal(w2[k], w3[k]) for k in w3), "iteration 3 did not train"


def test_independent_mode_builds_model_from_config(
    tmp_path, configs_dir, fake_openai, tiny_model_dir, tiny_tokenizer
) -> None:
    from pto.trainer import train

    cfg = _config(
        configs_dir, tmp_path / "runs",
        **{
            "tree.mode": "independent",
            "lookahead.k": 0,
            "models.policy": str(tiny_model_dir),
            "models.tokenizer": str(tiny_model_dir),
            "models.dtype": "auto",  # float32 on CPU
        },
    )
    assert cfg.run_name.endswith("_K0_MCL12_M3_indep")
    client = fake_openai()
    run_dir = train(cfg, client=client)

    it1 = run_dir / "iteration_1"
    pairs = _check_pairs(it1 / "pref_pairs", tiny_tokenizer, cfg.dpo.max_prompt_length)
    # Branch points: patient turns with >= MCL utterances so far and a recorded therapist reply.
    assert set(pairs["branch_turn_index"]) <= {11, 13, 15}
    _check_adapter(it1)
    _check_conversations(run_dir / "conversations" / "model_iter_0", seed=cfg.seed + 1)
    _check_conversations(run_dir / "conversations" / "model_iter_1", seed=cfg.seed + 2)
    # K = 0 and recorded conversations: patient calls only come from conversation generation.
    assert len(client.patient_calls) <= 2 * N_CONVS * (LENGTH // 2)
