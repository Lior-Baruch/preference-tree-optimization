"""Iterative PTO training: generate -> build preference pairs -> DPO -> save, repeated.

Iteration ``n`` (1-based):

1. Generate ``loop.conversations_per_iteration`` conversations with the current policy (model
   state ``n - 1``) against the simulated patient. They are saved to
   ``{run_dir}/conversations/model_iter_{n-1}/`` and double as that model state's evaluation set.
2. Build preference pairs (:mod:`pto.tree`, greedy or independent mode).
3. Train ``DPOTrainer`` on the pairs for ``loop.epochs_per_iteration`` epochs. LoRA weights carry
   over between iterations; the optimizer is fresh each iteration (warm restart).
4. Save the adapter to ``iteration_{n}/adapter`` (this directory marks the iteration as done).

After the last iteration a final conversation set is generated with the final policy
(``model_iter_{N}``). Every phase resumes after a crash: conversations from their CSVs, the pair
build from ``pairs.csv`` / ``_progress.json``, DPO from the newest valid HF checkpoint, and whole
iterations from their ``adapter`` directories.

Run layout::

    {output_dir}/{run_name}/
      config.yaml  run_metadata.json
      conversations/model_iter_{n}/conversation_{i}.csv  manifest.json  ({"i": persona_id})
      iteration_{n}/adapter/  training/  pref_pairs/  eda/  iteration_metadata.json
"""

from __future__ import annotations

import gc
import importlib.util
import json
import logging
import os
import random
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from . import _compat  # noqa: F401

# isort: split
import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

from . import __version__
from .clients import hf_login, make_openai_client
from .config import PTORunConfig, dump_config
from .conversation import ConversationState
from .generation import generate_all_conversations, run_async
from .lookahead import LookaheadConfig
from .model import (
    ADAPTER_SUBDIR,
    CHATML_TEMPLATE,
    ITER_PREFIX,
    build_lora_config,
    build_quantization_config,
    compute_cumulative_step_offset,
    get_adapter_param_count,
    list_hf_checkpoints,
    list_iteration_checkpoints,
    load_base_model,
    patch_generate,
    resolve_start_state,
    setup_permutations,
    setup_tokenizer,
    sync_pad_token,
)
from .oracle import OracleAsyncPrimitives, OracleConfig
from .tracking import (
    CheckpointMetadataCallback,
    GenerationRecorder,
    build_wandb_ctx,
    finish_iteration_logging,
    finish_wandb,
    init_iteration_logging,
    log_session,
    metadata_fields,
    patch_trainer_tensorboard_callback,
    setup_tensorboard_logging,
)
from .tree import (
    PROGRESS_FILENAME,
    _remove_progress,
    _slice_prefix_seeds,
    extract_pref_pairs_from_conversations,
    grow_preference_trees_batch,
    load_completed_pairs,
    save_completed_pairs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_iteration_datasets_dpo",
    "generate",
    "run_final_eval",
    "run_generation_only",
    "run_one_iteration",
    "run_training_phase",
    "save_iteration_checkpoint",
    "select_personas",
    "train",
    "write_run_metadata",
]

MANIFEST_FILENAME = "manifest.json"


# ── Small helpers ────────────────────────────────────────────────────────────────


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _conv_dir_for_iter(cfg: PTORunConfig, model_iter_label: int) -> str:
    """``{run_dir}/conversations/model_iter_{label}``."""
    return str(cfg.run_dir / "conversations" / f"model_iter_{model_iter_label}")


def _resolve_bf16(setting: Any) -> bool:
    """``dpo.bf16``: ``auto`` = CUDA with bf16 support; an explicit bool is used as-is."""
    if setting == "auto":
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    return bool(setting)


_REPORTING_BACKENDS: dict[str, tuple[str, ...]] = {
    # report_to name -> modules transformers can use for it (any one suffices)
    "tensorboard": ("tensorboard", "tensorboardX"),
    "wandb": ("wandb",),
}


def _check_reporting_backends(report_to: Sequence[str]) -> None:
    """Fail before any generation or API spend if a ``logging.report_to`` backend is missing.

    Otherwise transformers only raises when the first DPO trainer is built, i.e. after the
    iteration's conversations and preference trees have already been paid for.
    """
    for name in report_to:
        modules = _REPORTING_BACKENDS.get(name)
        if modules and not any(importlib.util.find_spec(m) is not None for m in modules):
            raise ImportError(
                f"logging.report_to includes {name!r} but {modules[0]!r} is not installed; "
                "install the 'tracking' extra (pip install -e \".[tracking]\", or pip install "
                f"{modules[0]}) or remove it from logging.report_to."
            )


def _prepare_injected_tokenizer(tokenizer: Any) -> Any:
    """Give an injected tokenizer the settings :func:`pto.model.setup_tokenizer` would apply
    when it lacks them (chat template, pad token, left truncation)."""
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = CHATML_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return tokenizer


def select_personas(
    all_permutations: Sequence[dict[str, Any]], seed: int, n: int
) -> tuple[list[dict[str, Any]], list[int]]:
    """Shuffle the persona grid with ``random.Random(seed)`` and take the first ``n``.

    Returns ``(personas, persona_ids)``; conversation ``i`` uses ``personas[i]`` whose grid index
    is ``persona_ids[i]``. Shuffling the index list consumes the RNG exactly like shuffling the
    persona list itself, so the order matches the reported runs (iteration ``n`` uses
    ``seed + n``; the final set uses ``seed + num_iterations + 1``).
    """
    order = list(range(len(all_permutations)))
    random.Random(seed).shuffle(order)
    ids = order[:n]
    return [all_permutations[i] for i in ids], ids


def _write_manifest(conv_dir: str, persona_ids: Sequence[int]) -> None:
    """``manifest.json``: ``{"<i>": persona_id}`` so evaluation can pair runs by persona."""
    os.makedirs(conv_dir, exist_ok=True)
    path = os.path.join(conv_dir, MANIFEST_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({str(i): int(pid) for i, pid in enumerate(persona_ids)}, f, indent=2)
    os.replace(tmp, path)


# ── Run metadata ─────────────────────────────────────────────────────────────────


def write_run_metadata(cfg: PTORunConfig) -> str:
    """Write ``run_metadata.json`` and the resolved ``config.yaml`` into the run directory."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    payload = {
        "run_name": cfg.run_name,
        "method": "PTO",
        "package_version": __version__,
        "started_at": datetime.now().isoformat(),
        "config": cfg.to_dict(),
    }
    path = run_dir / "run_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Run metadata: %s", path)
    return str(path)


# ── Generation ───────────────────────────────────────────────────────────────────


def run_generation_only(
    *,
    policy: Any,
    tokenizer: Any,
    client: Any,
    active_permutations: Sequence[dict[str, Any]],
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    conv_dir: str | None,
    cfg: PTORunConfig,
    patient_api_seed: int | None = None,
    num_utterances: int | None = None,
    persona_ids: Sequence[int] | None = None,
) -> tuple[list[ConversationState], float, float]:
    """Generate one conversation per persona with the current policy.

    ``conv_dir=None`` skips persistence. ``num_utterances`` overrides
    ``loop.conversation_length``. With ``persona_ids`` (aligned with ``active_permutations``) a
    ``manifest.json`` is written next to the CSVs.

    Returns ``(completed_states, generation_seconds, mean_conversation_length)``.
    """
    gen_start = time.time()
    policy.eval()
    policy.config.use_cache = True
    if conv_dir is not None and persona_ids is not None:
        _write_manifest(conv_dir, persona_ids)

    g = cfg.generation
    completed_states = generate_all_conversations(
        therapist_model=policy,
        therapist_tokenizer=tokenizer,
        client=client,
        permutations=list(active_permutations),
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        save_dir=conv_dir,
        patient_model_id=cfg.models.patient,
        max_tokens_per_response=g.max_new_tokens,
        num_utterances=(
            num_utterances if num_utterances is not None else cfg.loop.conversation_length
        ),
        temperature_therapist=g.therapist_temperature,
        temperature_patient=g.patient_temperature,
        batch_size=g.batch_size,
        therapist_max_input_tokens=g.max_input_tokens,
        patient_api_concurrency=g.patient_concurrency,
        patient_api_max_retries=g.patient_max_retries,
        patient_api_seed=patient_api_seed,
        patient_api_backoff_seconds=g.patient_backoff_s,
        batch_cooldown_seconds=1.0,
        max_retries_without_progress=g.max_retries_without_progress,
        stop_strings=g.stop_strings,
        verbose=True,
        verbose_detailed=logger.isEnabledFor(logging.DEBUG),
    )

    gen_time = time.time() - gen_start
    avg_conv_len = (
        float(np.mean([len(s.conversation) for s in completed_states]))
        if completed_states
        else 0.0
    )
    logger.info(
        "Generated %d conversations in %.1fs (mean length %.1f utterances)",
        len(completed_states),
        gen_time,
        avg_conv_len,
    )
    return completed_states, gen_time, avg_conv_len


# ── Datasets + DPO ───────────────────────────────────────────────────────────────


def build_iteration_datasets_dpo(
    pref_pairs: list[dict[str, Any]],
    eval_split_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    """Train/eval datasets of (prompt, chosen, rejected), split by conversation (no leakage).

    Returns ``(train_dataset, eval_dataset)``. ``eval_dataset`` is ``None`` when fewer than two
    conversations produced pairs (nothing can be held out); DPO evaluation is then disabled.
    """
    if not pref_pairs:
        raise ValueError(
            "build_iteration_datasets_dpo received 0 preference pairs: every branch tied within "
            "tree.tau, or tree.min_conv_length filtered every branch point. Lower tree.tau / "
            "tree.min_conv_length or raise tree.branches."
        )

    conv_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for p in pref_pairs:
        conv_groups[p["conversation_id"]].append(p)

    conv_ids = sorted(conv_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(conv_ids)

    # Keep >= 1 conversation in train even when very few conversations yield pairs.
    n_eval_convs = (
        min(max(1, int(len(conv_ids) * eval_split_ratio)), len(conv_ids) - 1)
        if len(conv_ids) >= 2
        else 0
    )
    eval_conv_ids = set(conv_ids[:n_eval_convs])
    train_conv_ids = set(conv_ids[n_eval_convs:])

    train_pairs = [p for cid in train_conv_ids for p in conv_groups[cid]]
    eval_pairs = [p for cid in eval_conv_ids for p in conv_groups[cid]]
    if not train_pairs:
        raise ValueError(
            f"All {len(pref_pairs)} preference pairs landed in eval (train empty): "
            f"conversations={len(conv_ids)}, eval conversations={n_eval_convs}. Raise "
            "loop.conversations_per_iteration or lower dpo.eval_split."
        )

    def _to_dataset(pairs: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_dict(
            {
                "prompt": [p["prompt"] for p in pairs],
                "chosen": [p["chosen"] for p in pairs],
                "rejected": [p["rejected"] for p in pairs],
            }
        )

    logger.info(
        "Grouped split: %d train / %d eval conversations; %d train / %d eval pairs",
        len(train_conv_ids),
        len(eval_conv_ids),
        len(train_pairs),
        len(eval_pairs),
    )
    if not eval_pairs:
        logger.warning(
            "Only %d conversation(s) produced pairs; no eval split this iteration "
            "(DPO evaluation disabled)",
            len(conv_ids),
        )
        return _to_dataset(train_pairs), None
    return _to_dataset(train_pairs), _to_dataset(eval_pairs)


def _build_dpo_args(
    cfg: PTORunConfig, inner_outdir: str, num_train_pairs: int, has_eval: bool = True
) -> DPOConfig:
    """``DPOConfig`` for one iteration."""
    d = cfg.dpo
    effective_batch_size = d.per_device_train_batch_size * d.gradient_accumulation_steps
    steps_per_epoch = max(1, int(np.ceil(num_train_pairs / effective_batch_size)))
    total_train_steps = max(1, steps_per_epoch * cfg.loop.epochs_per_iteration)
    warmup_steps = max(0, int(np.ceil(total_train_steps * d.warmup_ratio)))
    logger.info("DPO: %d optimizer steps, %d warmup", total_train_steps, warmup_steps)

    os.makedirs(os.path.join(inner_outdir, "tb_logs"), exist_ok=True)

    return DPOConfig(
        output_dir=inner_outdir,
        hub_model_id=(cfg.hub.repo_id if cfg.hub.push else None),
        run_name=cfg.run_name,
        per_device_train_batch_size=d.per_device_train_batch_size,
        per_device_eval_batch_size=d.per_device_eval_batch_size,
        gradient_accumulation_steps=d.gradient_accumulation_steps,
        learning_rate=d.learning_rate,
        num_train_epochs=cfg.loop.epochs_per_iteration,
        # TRL caps the full tokenized prompt + completion with a single max_length and truncates
        # with keep_start (drops the END on overflow). The pair builders already cap the prompt
        # to max_prompt_length (build_truncated_training_prompt), so the response is never cut.
        max_length=d.max_prompt_length + d.max_completion_length,
        beta=d.beta,
        loss_type=d.loss_type,
        bf16=_resolve_bf16(d.bf16),
        # Reference log-probs in a no-grad pre-pass: equivalent for DPO (the reference is
        # frozen), and the training step then needs no reference forward / adapter switch,
        # which frees memory during the step.
        precompute_ref_log_probs=d.precompute_ref_log_probs,
        # Recompute activations in backward: a large memory cut for full-sequence DPO logits.
        # use_reentrant=False is the mode that works with PEFT.
        gradient_checkpointing=d.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg.seed,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=cfg.logging.logging_steps,
        report_to=(list(cfg.logging.report_to) or "none"),
        # Step checkpoints keep a crash cheap; resume walks back to the newest valid one.
        save_strategy="steps",
        save_steps=cfg.logging.save_steps,
        save_total_limit=cfg.logging.save_total_limit,
        push_to_hub=False,
        eval_strategy=("epoch" if has_eval else "no"),
    )


def run_training_phase(
    *,
    policy: Any,
    tokenizer: Any,
    dpo_args: DPOConfig,
    lora_config: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    iteration: int,
    start_iteration: int,
    resume_checkpoint: str | None,
    cumulative_step_offset: int,
    iter_metadata_base: dict[str, Any],
    wandb_ctx: dict[str, Any] | None,
    report_to: list[str],
    tensorboard_log_dir: str | None = None,
) -> tuple[Any, int, float]:
    """Train one iteration's ``DPOTrainer``.

    LoRA is attached on the first iteration only (later iterations train the existing adapter).
    An HF checkpoint is resumed only for the iteration the run restarted in.

    Returns ``(updated_policy, global_step_delta, train_seconds)``.
    """
    train_start = time.time()
    policy.config.use_cache = False
    policy.train()

    is_already_peft = isinstance(policy, PeftModel) or hasattr(policy, "peft_config")
    peft_cfg = None if is_already_peft else lora_config

    setup_tensorboard_logging(report_to, tensorboard_log_dir)
    init_iteration_logging(report_to, iteration, cumulative_step_offset, wandb_ctx)

    trainer = DPOTrainer(
        model=policy,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_cfg,
        callbacks=[CheckpointMetadataCallback(iteration=iteration, metadata=iter_metadata_base)],
    )

    if "tensorboard" in report_to and tensorboard_log_dir is not None:
        patch_trainer_tensorboard_callback(trainer, tensorboard_log_dir)

    # Wrapping replaces generate(); re-inject the tokenizer so stop_strings keep working.
    patch_generate(trainer.model, tokenizer)

    resume = resume_checkpoint if (iteration == start_iteration and resume_checkpoint) else None
    if resume:
        logger.info("Resuming DPO from HF checkpoint %s", os.path.basename(resume))
    trainer.train(resume_from_checkpoint=resume)

    patch_generate(trainer.model, tokenizer)

    updated_policy = trainer.model
    resumed_steps = int(os.path.basename(resume).split("-")[-1]) if resume else 0
    step_delta = max(0, trainer.state.global_step - resumed_steps)
    train_time = time.time() - train_start

    del trainer
    logger.info("DPO training done in %.1fs", train_time)
    return updated_policy, step_delta, train_time


# ── Checkpoint save + iteration driver ───────────────────────────────────────────


def save_iteration_checkpoint(
    *,
    policy: Any,
    tokenizer: Any,
    iter_dir: str,
    inner_outdir: str,
    iteration: int,
    num_iterations: int,
    num_conversations: int,
    num_pref_pairs: int,
    avg_conv_len: float,
    gen_time: float,
    train_time: float,
    iter_metadata: dict[str, Any],
    report_to: list[str],
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    """Save the adapter (+ tokenizer + metadata) to ``iteration_N/adapter``; optional Hub push.

    The existence of ``iteration_N/adapter`` marks the iteration as complete for resume.
    """
    adapter_save_path = os.path.join(iter_dir, ADAPTER_SUBDIR)
    # Save ONLY the policy adapter, not the frozen "ref" adapter TRL adds for PEFT DPO (a copy of
    # the iteration-start weights used only as the reference). Resume loads the root "default"
    # adapter. "ref" is deleted from the live model at the end of this function, and the next
    # DPOTrainer re-creates it from the current weights.
    save_adapters = (
        ["default"]
        if (isinstance(policy, PeftModel) and "ref" in getattr(policy, "peft_config", {}))
        else None
    )
    policy.save_pretrained(adapter_save_path, selected_adapters=save_adapters)
    tokenizer.save_pretrained(adapter_save_path)
    logger.info("Adapter saved: %s", adapter_save_path)

    saved_ckpts = list_hf_checkpoints(inner_outdir)
    iter_metadata["epoch_checkpoints"] = [os.path.basename(c) for c in saved_ckpts]
    with open(os.path.join(iter_dir, "iteration_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(iter_metadata, f, indent=2)

    if "tensorboard" in report_to:
        tb_dir = Path(inner_outdir, "tb_logs")
        if not (tb_dir.is_dir() and any(tb_dir.rglob("events.out.tfevents.*"))):
            logger.warning("TensorBoard: no event files found in %s", tb_dir)

    if push_to_hub and hub_repo_id:
        try:
            policy.push_to_hub(
                repo_id=hub_repo_id,
                commit_message=f"iteration_{iteration}/{num_iterations}",
            )
            logger.info("Adapter pushed to %s (iteration %d/%d)", hub_repo_id, iteration,
                        num_iterations)
        except Exception as e:  # a failed push must not lose the local adapter
            logger.warning(
                "Hub push failed for iteration %d (%s: %s); local copy intact at %s",
                iteration, type(e).__name__, e, adapter_save_path,
            )

    # Drop TRL's frozen reference copy from the live model; the next DPOTrainer re-creates it from
    # the current weights (pi_{n-1}). peft >= 0.20 raises on a duplicate "ref" adapter otherwise.
    # Done after the optional Hub push so the pushed files are unchanged.
    if save_adapters is not None:
        policy.delete_adapter("ref")
        policy.set_adapter("default")

    finish_iteration_logging(
        report_to,
        iteration,
        {
            "iteration/num": iteration,
            "iteration/num_conversations": num_conversations,
            "iteration/num_pref_pairs": num_pref_pairs,
            "iteration/avg_conv_length": avg_conv_len,
            "iteration/generation_time_s": gen_time,
            "iteration/training_time_s": train_time,
            **(extra_metrics or {}),
        },
    )


def run_one_iteration(
    *,
    iteration: int,
    start_iteration: int,
    resume_checkpoint: str | None,
    cumulative_step_offset: int,
    policy: Any,
    tokenizer: Any,
    client: Any,
    all_permutations: Sequence[dict[str, Any]],
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    lora_config: Any,
    oracle_cfg: OracleConfig,
    lookahead_cfg: LookaheadConfig,
    primitives: OracleAsyncPrimitives,
    wandb_ctx: dict[str, Any] | None,
    cfg: PTORunConfig,
) -> tuple[Any, int, int, int]:
    """One PTO iteration: generate -> build pairs -> DPO -> save.

    Returns ``(new_policy, step_delta, num_conversations, num_pref_pairs)``.
    """
    iter_start_time = time.time()
    iter_dir = str(cfg.run_dir / f"{ITER_PREFIX}{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    conv_dir = _conv_dir_for_iter(cfg, iteration - 1)

    # Candidate-level records, kept in memory and flushed once after the pair build.
    recorder = GenerationRecorder(
        os.path.join(iter_dir, "eda", "generations.jsonl"),
        enabled=cfg.logging.save_generations,
        save_transcripts=cfg.logging.save_lookahead_transcripts,
    )

    active_permutations, persona_ids = select_personas(
        all_permutations, cfg.seed + iteration, cfg.loop.conversations_per_iteration
    )

    logger.info(
        "=== PTO iteration %d/%d (conversations from model_iter_%d) ===",
        iteration, cfg.loop.num_iterations, iteration - 1,
    )

    # Step 1: conversations (resumes from existing CSVs).
    logger.info("Step 1: generating %d conversations", len(active_permutations))
    completed_states, gen_time, avg_conv_len = run_generation_only(
        policy=policy,
        tokenizer=tokenizer,
        client=client,
        active_permutations=active_permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        conv_dir=conv_dir,
        cfg=cfg,
        patient_api_seed=cfg.seed + iteration,
        persona_ids=persona_ids,
    )
    _free_memory()

    # Step 2: preference pairs (resume-aware; see pto.tree).
    tc = cfg.tree
    logger.info(
        "Step 2: building preference pairs (mode=%s, M=%d, tau=%s, MCL=%d, K=%d)",
        tc.mode, tc.branches, tc.tau, tc.min_conv_length, cfg.lookahead.k,
    )
    pref_dir = os.path.join(iter_dir, "pref_pairs")
    os.makedirs(pref_dir, exist_ok=True)
    progress_path = os.path.join(pref_dir, PROGRESS_FILENAME)
    pref_start = time.time()

    pref_pairs = load_completed_pairs(pref_dir, cfg)
    if pref_pairs is not None:
        # A completed build is on disk. Do NOT flush the recorder: generations.jsonl was written
        # by that build and the fresh recorder is empty.
        pref_time = time.time() - pref_start
        logger.info("Reloaded %d preference pairs from pairs.csv; skipping the build",
                    len(pref_pairs))
    else:
        if tc.mode == "greedy":
            # Seeds are the first MCL utterances of the Step-1 conversations; those
            # conversations themselves stay untouched as the evaluation data.
            seed_states = _slice_prefix_seeds(completed_states, tc.min_conv_length)
            logger.info(
                "Sliced %d tree seeds (%d utterances, ending on a patient turn) from %d "
                "conversations", len(seed_states), tc.min_conv_length, len(completed_states),
            )
            _free_memory()
            pref_pairs = run_async(
                grow_preference_trees_batch(
                    seed_states=seed_states,
                    permutations=active_permutations,
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
                    progress_path=progress_path,
                )
            )
        else:
            pref_pairs = run_async(
                extract_pref_pairs_from_conversations(
                    completed_states=completed_states,
                    permutations=active_permutations,
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
                    progress_path=progress_path,
                )
            )
        pref_time = time.time() - pref_start
        logger.info("Built %d preference pairs in %.1fs from %d conversations",
                    len(pref_pairs), pref_time, len(completed_states))

        pairs_csv = save_completed_pairs(pref_pairs, pref_dir, cfg)
        logger.info("Preference pairs saved: %s", pairs_csv)
        if recorder.flush() is not None:
            logger.info("Generation records saved: %s (%d branch points)",
                        recorder.out_path, len(recorder.records))
        _remove_progress(progress_path)

    # Fail fast with guidance; the (empty) pairs.csv is kept as evidence.
    if not pref_pairs:
        raise ValueError(
            f"Iteration {iteration} produced 0 preference pairs from {len(completed_states)} "
            f"conversations (mean length {avg_conv_len:.1f}). Every branch tied within "
            f"tree.tau={tc.tau}, or tree.min_conv_length={tc.min_conv_length} filtered every "
            f"branch point. Lower tree.tau / tree.min_conv_length, or raise tree.branches "
            f"(currently {tc.branches})."
        )
    _free_memory()

    # Step 3: datasets.
    train_dataset, eval_dataset = build_iteration_datasets_dpo(
        pref_pairs=pref_pairs, eval_split_ratio=cfg.dpo.eval_split, seed=cfg.seed
    )

    # Step 4: DPO.
    logger.info("Step 3: DPO for %d epoch(s)", cfg.loop.epochs_per_iteration)
    inner_outdir = os.path.join(iter_dir, "training")
    dpo_args = _build_dpo_args(
        cfg, inner_outdir, num_train_pairs=len(train_dataset), has_eval=eval_dataset is not None
    )
    iter_metadata_base = {
        "run_name": cfg.run_name,
        "method": "PTO",
        "iteration": iteration,
        "base_model": cfg.models.policy,
        "oracle_model": cfg.models.oracle,
        "questionnaire_ids": list(cfg.oracle.questionnaires),
        "tree_mode": tc.mode,
        "lookahead_k": cfg.lookahead.k,
        "min_conv_length": tc.min_conv_length,
        "num_branches_per_turn": tc.branches,
        "pref_filter_tau": tc.tau,
        "dpo_beta": cfg.dpo.beta,
        "dpo_loss_type": cfg.dpo.loss_type,
        "learning_rate": cfg.dpo.learning_rate,
        "lora_r": cfg.lora.r,
    }
    new_policy, step_delta, train_time = run_training_phase(
        policy=policy,
        tokenizer=tokenizer,
        dpo_args=dpo_args,
        lora_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        iteration=iteration,
        start_iteration=start_iteration,
        resume_checkpoint=resume_checkpoint,
        cumulative_step_offset=cumulative_step_offset,
        iter_metadata_base=iter_metadata_base,
        wandb_ctx=wandb_ctx,
        report_to=list(cfg.logging.report_to),
        tensorboard_log_dir=os.path.join(inner_outdir, "tb_logs"),
    )

    # Step 5: save. Append this process's timing first so the cumulative fields include it.
    log_session(iter_dir, generation_s=gen_time, training_s=train_time, pref_pair_s=pref_time,
                started_at=iter_start_time)
    iter_metadata = {
        **iter_metadata_base,
        "num_conversations": len(completed_states),
        "num_pref_pairs": len(pref_pairs),
        "avg_conversation_length": float(avg_conv_len),
        "epochs_per_iteration": cfg.loop.epochs_per_iteration,
        "generation_time_s": gen_time,
        "pref_pair_time_s": pref_time,
        "training_time_s": train_time,
        **metadata_fields(iter_dir),
    }
    save_iteration_checkpoint(
        policy=new_policy,
        tokenizer=tokenizer,
        iter_dir=iter_dir,
        inner_outdir=inner_outdir,
        iteration=iteration,
        num_iterations=cfg.loop.num_iterations,
        num_conversations=len(completed_states),
        num_pref_pairs=len(pref_pairs),
        avg_conv_len=avg_conv_len,
        gen_time=gen_time,
        train_time=train_time,
        iter_metadata=iter_metadata,
        report_to=list(cfg.logging.report_to),
        push_to_hub=cfg.hub.push,
        hub_repo_id=cfg.hub.repo_id,
        extra_metrics=(recorder.aggregate()[0] if recorder.records else None),
    )

    logger.info(
        "Iteration %d done in %.1fs: %d conversations, %d preference pairs",
        iteration, time.time() - iter_start_time, len(completed_states), len(pref_pairs),
    )
    return new_policy, step_delta, len(completed_states), len(pref_pairs)


def run_final_eval(
    *,
    policy: Any,
    tokenizer: Any,
    client: Any,
    all_permutations: Sequence[dict[str, Any]],
    therapist_system_prompt: str,
    therapist_init_utterance: str,
    cfg: PTORunConfig,
) -> str:
    """Generate the final policy's conversation set (``model_iter_{num_iterations}``)."""
    final_label = cfg.loop.num_iterations
    final_conv_dir = _conv_dir_for_iter(cfg, final_label)
    logger.info("=== Final conversations: model_iter_%d ===", final_label)

    final_seed = cfg.seed + cfg.loop.num_iterations + 1
    final_active, persona_ids = select_personas(
        all_permutations, final_seed, cfg.loop.conversations_per_iteration
    )
    final_states, final_gen_time, final_avg_len = run_generation_only(
        policy=policy,
        tokenizer=tokenizer,
        client=client,
        active_permutations=final_active,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        conv_dir=final_conv_dir,
        cfg=cfg,
        patient_api_seed=final_seed,
        persona_ids=persona_ids,
    )
    logger.info(
        "Final set: %d conversations in %.1fs (mean length %.1f), saved to %s",
        len(final_states), final_gen_time, final_avg_len, final_conv_dir,
    )
    del final_states
    _free_memory()
    return final_conv_dir


# ── Public entry points ──────────────────────────────────────────────────────────


def _load_policy_and_tokenizer(
    cfg: PTORunConfig, *, tokenizer: Any, model: Any, for_training: bool
) -> tuple[Any, Any]:
    """Build (or adopt injected) tokenizer and base model; sync pad ids, patch ``generate``."""
    if tokenizer is None or model is None:
        hf_login()
    if tokenizer is None:
        tokenizer = setup_tokenizer(cfg.tokenizer_id)
    else:
        tokenizer = _prepare_injected_tokenizer(tokenizer)
    if model is None:
        quant_cfg = build_quantization_config() if cfg.models.load_in_4bit else None
        model = load_base_model(
            cfg.models.policy, quant_cfg, for_training=for_training, dtype=cfg.models.dtype
        )
    else:
        model.config.use_cache = not for_training
    sync_pad_token(model, tokenizer)
    patch_generate(model, tokenizer)
    return tokenizer, model


def train(
    cfg: PTORunConfig,
    client: Any = None,
    tokenizer: Any = None,
    model: Any = None,
) -> Path:
    """Run (or resume) the full iterative PTO loop described by ``cfg``; returns the run dir.

    ``client`` (an ``openai.AsyncOpenAI``-compatible object), ``tokenizer`` and ``model`` (the
    base causal LM, without adapters) can be injected, e.g. for offline tests; otherwise they are
    built from ``cfg`` (``OPENAI_API_KEY`` / ``HF_TOKEN`` from the environment).
    """
    cfg.validate()
    _check_reporting_backends(cfg.logging.report_to)
    _seed_everything(cfg.seed)
    if client is None:
        # Before the run directory exists, so a missing API key leaves nothing behind.
        client = make_openai_client(base_url=cfg.models.api_base_url)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run %s -> %s", cfg.run_name, run_dir)
    if cfg.hub.push and tokenizer is not None and model is not None:
        hf_login()  # the push needs a token even when nothing is downloaded

    # Persona grid (persona_id = grid index) and the fixed expert-therapist prompt. The RNG is
    # re-seeded right before, as in the reported runs.
    random.seed(cfg.seed)
    all_permutations, therapist_system_prompt, therapist_init_utterance = setup_permutations(
        only_expert_therapist=True
    )
    if cfg.loop.conversations_per_iteration > len(all_permutations):
        raise ValueError(
            f"loop.conversations_per_iteration={cfg.loop.conversations_per_iteration} but only "
            f"{len(all_permutations)} personas exist"
        )

    lora_config = build_lora_config(
        cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout, cfg.lora.target_modules
    )
    tokenizer, base_policy = _load_policy_and_tokenizer(
        cfg, tokenizer=tokenizer, model=model, for_training=True
    )
    n_params = get_adapter_param_count(base_policy)
    logger.info(
        "Policy %s loaded: %.1fM parameters (%.2fM trainable before LoRA)",
        cfg.models.policy, n_params["total_M"], n_params["trainable_M"],
    )

    oracle_cfg = OracleConfig(
        model_id=cfg.models.oracle,
        request_timeout=cfg.oracle.timeout_s,
        max_retries=cfg.oracle.max_retries,
        eval_temperature=cfg.oracle.temperature,
        max_concurrency=cfg.oracle.concurrency,
    )
    lookahead_cfg = LookaheadConfig(
        k=cfg.lookahead.k,
        temperature_therapist=cfg.lookahead.therapist_temperature,
        temperature_patient=cfg.lookahead.patient_temperature,
        max_tokens=cfg.lookahead.max_new_tokens,
        max_input_tokens=cfg.generation.max_input_tokens,
        patient_model_id=cfg.models.patient,
        patient_api_concurrency=cfg.generation.patient_concurrency,
        patient_api_max_retries=cfg.generation.patient_max_retries,
        patient_api_backoff_seconds=cfg.generation.patient_backoff_s,
        stop_strings=list(cfg.generation.stop_strings),
        lookahead_sub_batch_size=cfg.lookahead.sub_batch_size,
    )
    primitives = OracleAsyncPrimitives(
        oracle_concurrency=cfg.oracle.concurrency,
        lookahead_patient_concurrency=cfg.generation.patient_concurrency,
    )

    wandb_ctx = build_wandb_ctx(cfg)
    write_run_metadata(cfg)

    start_iteration, policy, resume_checkpoint = resolve_start_state(
        str(run_dir), base_policy, tokenizer
    )
    cumulative_step_offset = compute_cumulative_step_offset(str(run_dir))
    if start_iteration > cfg.loop.num_iterations:
        logger.info("All %d iterations are already trained", cfg.loop.num_iterations)
    else:
        logger.info("Starting at iteration %d of %d", start_iteration, cfg.loop.num_iterations)

    # Each iteration builds a new DPOTrainer: LoRA weights carry over, Adam state resets.
    iterations_run = []
    for iteration in range(start_iteration, cfg.loop.num_iterations + 1):
        policy, step_delta, _n_convs, _n_pairs = run_one_iteration(
            iteration=iteration,
            start_iteration=start_iteration,
            resume_checkpoint=resume_checkpoint,
            cumulative_step_offset=cumulative_step_offset,
            policy=policy,
            tokenizer=tokenizer,
            client=client,
            all_permutations=all_permutations,
            therapist_system_prompt=therapist_system_prompt,
            therapist_init_utterance=therapist_init_utterance,
            lora_config=lora_config,
            oracle_cfg=oracle_cfg,
            lookahead_cfg=lookahead_cfg,
            primitives=primitives,
            wandb_ctx=wandb_ctx,
            cfg=cfg,
        )
        cumulative_step_offset += step_delta
        resume_checkpoint = None
        iterations_run.append(iteration)
        _free_memory()

    final_conv_dir = run_final_eval(
        policy=policy,
        tokenizer=tokenizer,
        client=client,
        all_permutations=all_permutations,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        cfg=cfg,
    )

    if cfg.hub.push and cfg.hub.repo_id:
        try:
            policy.push_to_hub(
                repo_id=cfg.hub.repo_id,
                commit_message=f"PTO training complete: {cfg.loop.num_iterations} iterations",
            )
            logger.info("Final adapter pushed to %s", cfg.hub.repo_id)
        except Exception as e:  # the local adapters are the source of truth
            logger.warning("Final Hub push failed (%s: %s)", type(e).__name__, e)

    finish_wandb(list(cfg.logging.report_to))

    saved = [n for n, _ in list_iteration_checkpoints(str(run_dir))]
    logger.info(
        "PTO training complete: %d iteration(s) this call, adapters for iterations %s, "
        "final conversations in %s",
        len(iterations_run), saved, final_conv_dir,
    )
    return run_dir


def generate(
    cfg: PTORunConfig,
    adapter: str | None,
    out_dir: str | os.PathLike[str],
    num_conversations: int | None = None,
    seed: int | None = None,
    client: Any = None,
    tokenizer: Any = None,
    model: Any = None,
) -> Path:
    """Generate a conversation set with the base policy or a trained adapter.

    Args:
        cfg: Run config (models, generation settings, conversation length).
        adapter: Path or Hub id of a LoRA adapter (e.g. ``{run_dir}/iteration_3/adapter``), or
            ``None`` / ``"base"`` for the untrained base model.
        out_dir: Destination for ``conversation_{i}.csv`` and ``manifest.json``. Existing CSVs
            are kept and only missing conversations are generated.
        num_conversations: Defaults to ``loop.conversations_per_iteration``.
        seed: Persona-shuffle and patient-API seed (default ``cfg.seed``). The training run
            used ``seed + n + 1`` for ``model_iter_{n}``.
        client, tokenizer, model: Optional injected objects, as in :func:`train`.

    Returns:
        ``Path(out_dir)``.
    """
    seed = cfg.seed if seed is None else int(seed)
    n = num_conversations or cfg.loop.conversations_per_iteration
    _seed_everything(seed)
    if client is None:
        client = make_openai_client(base_url=cfg.models.api_base_url)

    random.seed(seed)
    all_permutations, therapist_system_prompt, therapist_init_utterance = setup_permutations(
        only_expert_therapist=True
    )
    if n > len(all_permutations):
        raise ValueError(f"num_conversations={n} but only {len(all_permutations)} personas exist")
    active, persona_ids = select_personas(all_permutations, seed, n)

    tokenizer, policy = _load_policy_and_tokenizer(
        cfg, tokenizer=tokenizer, model=model, for_training=False
    )
    use_adapter = adapter is not None and str(adapter).lower() != "base"
    if use_adapter:
        policy = PeftModel.from_pretrained(policy, str(adapter), is_trainable=False)
        patch_generate(policy, tokenizer)
        logger.info("Loaded adapter %s", adapter)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_generation_only(
        policy=policy,
        tokenizer=tokenizer,
        client=client,
        active_permutations=active,
        therapist_system_prompt=therapist_system_prompt,
        therapist_init_utterance=therapist_init_utterance,
        conv_dir=str(out),
        cfg=cfg,
        patient_api_seed=seed,
        persona_ids=persona_ids,
    )
    with open(out / "generation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": cfg.models.policy,
                "adapter": str(adapter) if use_adapter else "base",
                "patient": cfg.models.patient,
                "seed": seed,
                "num_conversations": n,
                "conversation_length": cfg.loop.conversation_length,
                "package_version": __version__,
                "created_at": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    return out
