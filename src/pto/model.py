"""Policy model, tokenizer, LoRA and on-disk checkpoint discovery.

Covers (a) loading the tokenizer and base policy (optionally 4-bit) and building the LoRA
config, (b) discovering what has already been checkpointed in a run directory, and
(c) resuming an interrupted multi-iteration run.

Heavy libraries (torch, transformers, peft) are imported inside the functions that need them
so that importing this module stays cheap; :mod:`pto._compat` is imported first each time to
keep the trl-before-torch import order.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import types
from typing import Any

logger = logging.getLogger(__name__)

# ── Run-directory layout ─────────────────────────────────────────────────────────
ITER_PREFIX = "iteration_"  # {run_dir}/iteration_3/
HF_CKPT_PREFIX = "checkpoint-"  # {run_dir}/iteration_3/training/checkpoint-500/
ADAPTER_SUBDIR = "adapter"  # {run_dir}/iteration_3/adapter/  (exists = iteration done)
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")
HF_TRAINER_FILES = ADAPTER_FILES + ("trainer_state.json",)

# ── Chat template ────────────────────────────────────────────────────────────────
# Plain ChatML. A base (non-instruct) Llama never saw these markers as special tokens, so
# they are ordinary BPE pieces: generation stops on them via ``stop_strings`` and
# ``pto.conversation.clean_completion`` strips any residue. Keep this template, the
# configured stop strings and the markers in ``pto.conversation`` in sync.
CHATML_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}"
    "{% set add_generation_prompt = false %}{% endif %}"
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)

_DTYPE_NAMES = ("auto", "bfloat16", "float16", "float32")


def _trl_before_torch() -> None:
    """Import trl ahead of torch (see :mod:`pto._compat`) before any lazy torch import.

    A call rather than an inline import so that import sorting cannot reorder it.
    """
    from . import _compat  # noqa: F401


# ── Tokenizer ────────────────────────────────────────────────────────────────────


def setup_tokenizer(tokenizer_id: str, padding_side: str = "left") -> Any:
    """Load a tokenizer configured for batched ChatML generation.

    Sets ``pad = eos``, left truncation (keep the most recent turns) and the ChatML
    template. Left padding is required for batched causal generation.
    """
    _trl_before_torch()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.chat_template = CHATML_TEMPLATE
    return tokenizer


# ── Precision / quantization / LoRA ──────────────────────────────────────────────


def resolve_dtype(dtype: Any = "auto") -> Any:
    """Map a dtype name to a ``torch.dtype``.

    ``"auto"`` (or ``None``) picks bfloat16 on CUDA devices that support it, float16 on other
    CUDA devices and float32 on CPU (half precision on CPU is slow and numerically fragile).
    A ``torch.dtype`` is returned unchanged.
    """
    _trl_before_torch()
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or dtype == "auto":
        if not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype not in _DTYPE_NAMES:
        raise ValueError(f"dtype must be one of {_DTYPE_NAMES}, got {dtype!r}")
    return getattr(torch, dtype)


def build_quantization_config(
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    compute_dtype: Any = None,
) -> Any:
    """Build a 4-bit ``BitsAndBytesConfig`` (requires the optional ``bitsandbytes`` package).

    Args:
        load_in_4bit: Whether to load in 4-bit.
        quant_type: Quantization type (``"nf4"`` by default).
        compute_dtype: Compute dtype; ``None`` selects bfloat16 when CUDA supports it and
            float16 otherwise.
    """
    _trl_before_torch()
    import torch
    from transformers import BitsAndBytesConfig

    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "4-bit loading requires the optional 'bitsandbytes' package "
            "(pip install bitsandbytes) and a CUDA GPU."
        )
    if compute_dtype is None:
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
    )


def build_lora_config(
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> Any:
    """Build the policy's LoRA config (``peft.LoraConfig``, causal LM, no bias training)."""
    _trl_before_torch()
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )


# ── Checkpoint discovery ─────────────────────────────────────────────────────────


def _list_numbered_dirs(
    root: str | os.PathLike[str],
    prefix: str,
    separator: str = "_",
) -> list[tuple[int, str]]:
    """Subdirectories of ``root`` named ``{prefix}{N}``, sorted ascending by ``N``.

    Entries whose suffix is not an int are skipped; a missing ``root`` yields ``[]``.
    """
    root = os.fspath(root)
    if not os.path.isdir(root):
        return []
    entries: list[tuple[int, str]] = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not (name.startswith(prefix) and os.path.isdir(full)):
            continue
        try:
            n = int(name[len(prefix):].split(separator, 1)[0])
        except ValueError:
            continue
        entries.append((n, full))
    entries.sort(key=lambda x: x[0])
    return entries


def list_iteration_checkpoints(output_dir: str | os.PathLike[str]) -> list[tuple[int, str]]:
    """``[(iteration, adapter_path), ...]`` for every ``iteration_N/adapter`` directory."""
    result: list[tuple[int, str]] = []
    for n, iter_dir in _list_numbered_dirs(output_dir, ITER_PREFIX):
        adapter_path = os.path.join(iter_dir, ADAPTER_SUBDIR)
        if os.path.isdir(adapter_path):
            result.append((n, adapter_path))
    return result


def get_latest_iteration(output_dir: str | os.PathLike[str]) -> int:
    """Highest completed iteration, or 0 (= base model, no adapter)."""
    checkpoints = list_iteration_checkpoints(output_dir)
    return checkpoints[-1][0] if checkpoints else 0


def list_hf_checkpoints(training_dir: str | os.PathLike[str]) -> list[str]:
    """Trainer ``checkpoint-N`` directories in ``training_dir``, sorted ascending by step."""
    return [
        path for _, path in _list_numbered_dirs(training_dir, HF_CKPT_PREFIX, separator="-")
    ]


def validate_hf_checkpoint(checkpoint_path: str | os.PathLike[str]) -> tuple[bool, list[str]]:
    """Check that a trainer checkpoint can be resumed from; returns ``(ok, missing_files)``."""
    missing = [
        f for f in HF_TRAINER_FILES if not os.path.exists(os.path.join(checkpoint_path, f))
    ]
    return len(missing) == 0, missing


def get_latest_valid_hf_checkpoint(training_dir: str | os.PathLike[str]) -> str | None:
    """Newest complete ``checkpoint-N`` in ``training_dir``, or ``None``.

    Walks newest to oldest and skips half-written checkpoints, so a crash during a checkpoint
    write falls back to the previous good one instead of discarding the whole iteration.
    Keep ``save_total_limit >= 2`` so that a fallback exists on disk.
    """
    for path in reversed(list_hf_checkpoints(training_dir)):
        if validate_hf_checkpoint(path)[0]:
            return path
    return None


# ── Model loading ────────────────────────────────────────────────────────────────


def cap_cuda_memory(margin_gb: float = 1.5, max_gb: float | None = None) -> float | None:
    """Cap this process's PyTorch CUDA allocations below the currently free VRAM.

    Call before loading a model. On some drivers (notably Windows with system-memory fallback
    enabled) an allocation beyond physical VRAM is not refused but spilled into shared system
    memory, which can stall or destabilize the machine instead of raising. With
    ``torch.cuda.set_per_process_memory_fraction`` the caching allocator raises a catchable
    ``OutOfMemoryError`` instead, and the generation code's OOM paths back off (halve the
    batch). Memory outside the caching allocator (CUDA context, cuBLAS workspaces) is not
    capped; ``margin_gb`` leaves room for it.

    The cap is ``free - margin_gb`` (optionally ``min``-ed with ``max_gb``), measured after
    the CUDA context exists. Harmless on large GPUs.

    Returns:
        The cap in GB, or ``None`` without CUDA.
    """
    _trl_before_torch()
    import torch

    if not torch.cuda.is_available():
        return None
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    cap = free - margin_gb * 2**30
    if max_gb is not None:
        cap = min(cap, max_gb * 2**30)
    if cap <= 0:
        raise RuntimeError(
            f"Only {free / 2**30:.1f} GB of VRAM free; refusing to start "
            f"(margin {margin_gb} GB). Close other GPU programs first."
        )
    torch.cuda.set_per_process_memory_fraction(cap / total)
    logger.info(
        "CUDA memory cap: %.1f GB for this process (%.1f GB free of %.1f GB, margin %s GB)",
        cap / 2**30, free / 2**30, total / 2**30, margin_gb,
    )
    return cap / 2**30


def load_base_model(
    model_id: str,
    quantization_config: Any = None,
    for_training: bool = False,
    trust_remote_code: bool = False,
    attn_implementation: str | None = "sdpa",
    dtype: Any = "auto",
    device_map: Any = "auto",
) -> Any:
    """Load the base causal LM, optionally 4-bit quantized.

    - ``quantization_config`` given: bitsandbytes 4-bit load; for training, the bnb compute
      dtype is used and ``prepare_model_for_kbit_training`` is applied.
    - ``quantization_config`` is ``None``: unquantized load in ``dtype``. On a large GPU this
      avoids the per-matmul dequantization overhead; LoRA then trains under autocast.

    Args:
        model_id: Hugging Face model id or local path.
        quantization_config: A ``BitsAndBytesConfig`` or ``None``.
        for_training: Sets ``use_cache=False`` (and prepares a quantized model for k-bit
            training). Otherwise the KV cache is enabled for inference.
        trust_remote_code: Forwarded to ``from_pretrained``. Off by default; standard
            architectures such as Llama do not need it.
        attn_implementation: Attention backend (``"sdpa"`` by default); ``None`` omits it.
        dtype: ``"auto"`` | ``"bfloat16"`` | ``"float16"`` | ``"float32"`` or a
            ``torch.dtype``; see :func:`resolve_dtype`. Ignored when quantized.
        device_map: Forwarded to ``from_pretrained``.
    """
    _trl_before_torch()
    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        if for_training:
            kwargs["dtype"] = quantization_config.bnb_4bit_compute_dtype
        precision = "4bit"
    else:
        kwargs["dtype"] = resolve_dtype(dtype)
        precision = str(kwargs["dtype"]).replace("torch.", "")

    from transformers import AutoModelForCausalLM

    logger.info(
        "Loading base model %s (for_training=%s, precision=%s)", model_id, for_training, precision
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.config.use_cache = not for_training

    if for_training and quantization_config is not None:
        from peft import prepare_model_for_kbit_training

        prepare_model_for_kbit_training(model)

    return model


def sync_pad_token(model: Any, tokenizer: Any) -> None:
    """Copy the tokenizer's pad/eos/bos ids into the model and generation configs."""
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id


def patch_generate(model: Any, tokenizer: Any) -> None:
    """Make ``model.generate`` inject ``tokenizer=`` (required by ``stop_strings``).

    A no-op if this model object is already patched. Wrapping a model (e.g.
    ``PeftModel.from_pretrained``, or a trainer re-wrapping its model) produces an object
    whose ``generate`` is unpatched, so call this again after every wrap.
    """
    if hasattr(model, "_generate_patched"):
        return

    if not hasattr(model, "_original_generate"):
        model._original_generate = model.generate

    _tokenizer = tokenizer

    def generate_with_tokenizer(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "tokenizer" not in kwargs:
            kwargs["tokenizer"] = _tokenizer
        # transformers warns when generation_config is passed together with generation
        # kwargs such as disable_compile; move it onto the config to keep the behaviour
        # without the warning.
        if "generation_config" in kwargs and "disable_compile" in kwargs:
            gen_cfg = kwargs.get("generation_config")
            if gen_cfg is not None:
                gen_cfg.disable_compile = kwargs.pop("disable_compile")
        return self._original_generate(*args, **kwargs)

    model.generate = types.MethodType(generate_with_tokenizer, model)
    model._generate_patched = True
    logger.debug("Patched generate() for %s", type(model).__name__)


def get_adapter_param_count(model: Any) -> dict[str, float]:
    """Trainable vs total parameter counts.

    Returns:
        Dict with ``trainable``, ``total``, ``trainable_pct``, ``trainable_M``, ``total_M``.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100 * trainable / total if total > 0 else 0,
        "trainable_M": trainable / 1e6,
        "total_M": total / 1e6,
    }


# ── Personas ─────────────────────────────────────────────────────────────────────


def setup_permutations(
    only_expert_therapist: bool = True,
) -> tuple[list[dict[str, Any]], str, str]:
    """Build the patient persona grid and the therapist prompt / opening utterance.

    The therapist uses the ``Good`` (expert) personality level. The order of the returned
    permutations is the grid order, so a permutation's index is its ``persona_id``.

    Returns:
        ``(permutations, therapist_system_prompt, therapist_init_utterance)``.
    """
    from .personas import CounselorPersonality, generate_all_permutations

    permutations = generate_all_permutations(only_expert_therapist=only_expert_therapist)

    good_level = CounselorPersonality.PersonalityLevel.Good
    therapist = CounselorPersonality.choose_random_therapist_name()
    therapist_system_prompt = CounselorPersonality.build_system_prompt(
        personality_level=good_level, name=therapist["name"]
    )
    therapist_init_utterance = CounselorPersonality.get_init_utterance(
        personality_level=good_level, name=therapist["name"]
    )
    return permutations, therapist_system_prompt, therapist_init_utterance


# ── Iteration resume ─────────────────────────────────────────────────────────────


def resolve_start_state(
    local_outdir: str | os.PathLike[str],
    base_policy: Any,
    tokenizer: Any,
) -> tuple[int, Any, str | None]:
    """Find the iteration to start at, load the matching policy and any resume checkpoint.

    Iteration 0 is the base model (no adapter); iterations 1+ are trained.

    - Fresh start: nothing on disk, start at iteration 1 with the base policy.
    - Mid-iteration crash: ``iteration_{N+1}/training/checkpoint-*`` exists but
      ``iteration_{N+1}/adapter`` does not. Resume training from the newest *valid*
      checkpoint; if none is valid, restart iteration ``N+1`` from iteration ``N``'s adapter
      (or the base model when ``N == 0``).
    - Between iterations: ``iteration_N/adapter`` exists, start at ``N+1`` from that adapter.

    Adapters are always attached to the plain base model (never nested PEFT wrappers).

    Returns:
        ``(start_iteration, policy, resume_checkpoint)``.
    """
    _trl_before_torch()
    from peft import PeftModel

    local_outdir = os.fspath(local_outdir)
    latest_iteration = get_latest_iteration(local_outdir)

    base_for_adapter = (
        base_policy.get_base_model()
        if hasattr(base_policy, "peft_config") and hasattr(base_policy, "get_base_model")
        else base_policy
    )

    candidate_iter = latest_iteration + 1
    candidate_training_dir = os.path.join(
        local_outdir, f"{ITER_PREFIX}{candidate_iter}", "training"
    )
    all_ckpts = list_hf_checkpoints(candidate_training_dir)

    if all_ckpts:  # incomplete iteration
        valid_ckpt = get_latest_valid_hf_checkpoint(candidate_training_dir)
        if valid_ckpt is not None:
            newest = os.path.basename(all_ckpts[-1])
            if os.path.basename(valid_ckpt) != newest:
                logger.warning(
                    "Newest checkpoint %s is incomplete; falling back to %s",
                    newest, os.path.basename(valid_ckpt),
                )
            logger.info(
                "Resuming %s%d from %s", ITER_PREFIX, candidate_iter, os.path.basename(valid_ckpt)
            )
            policy = PeftModel.from_pretrained(base_for_adapter, valid_ckpt, is_trainable=True)
            patch_generate(policy, tokenizer)
            return candidate_iter, policy, valid_ckpt
        logger.warning(
            "No valid checkpoint in %s%d/training (%d found, all incomplete); "
            "restarting that iteration from scratch",
            ITER_PREFIX, candidate_iter, len(all_ckpts),
        )
        if latest_iteration == 0:
            policy = base_for_adapter
        else:
            adapter_path = os.path.join(
                local_outdir, f"{ITER_PREFIX}{latest_iteration}", ADAPTER_SUBDIR
            )
            policy = PeftModel.from_pretrained(base_for_adapter, adapter_path, is_trainable=True)
        patch_generate(policy, tokenizer)
        return candidate_iter, policy, None

    if latest_iteration > 0:  # between iterations
        adapter_path = os.path.join(
            local_outdir, f"{ITER_PREFIX}{latest_iteration}", ADAPTER_SUBDIR
        )
        logger.info("Resuming: loading adapter from %s%d", ITER_PREFIX, latest_iteration)
        policy = PeftModel.from_pretrained(base_for_adapter, adapter_path, is_trainable=True)
        patch_generate(policy, tokenizer)
        return candidate_iter, policy, None

    return 1, base_for_adapter, None  # fresh start


def compute_cumulative_step_offset(local_outdir: str | os.PathLike[str]) -> int:
    """Total optimizer steps of completed and in-progress iterations.

    Each iteration's trainer restarts ``global_step`` at 0; adding this offset keeps the
    logged step axis continuous across iterations. The in-progress iteration contributes the
    steps of the checkpoint :func:`resolve_start_state` would resume from.
    """
    local_outdir = os.fspath(local_outdir)
    offset = 0
    completed = list_iteration_checkpoints(local_outdir)
    for it_num, _ in completed:
        training_dir = os.path.join(local_outdir, f"{ITER_PREFIX}{it_num}", "training")
        ckpts = list_hf_checkpoints(training_dir)
        if ckpts:
            offset += int(os.path.basename(ckpts[-1]).split("-")[-1])

    latest_completed = completed[-1][0] if completed else 0
    in_progress_training_dir = os.path.join(
        local_outdir, f"{ITER_PREFIX}{latest_completed + 1}", "training"
    )
    in_progress_ckpt = get_latest_valid_hf_checkpoint(in_progress_training_dir)
    if in_progress_ckpt is not None:
        partial_steps = int(os.path.basename(in_progress_ckpt).split("-")[-1])
        offset += partial_steps
        logger.info(
            "Including %d steps from in-progress %s%d",
            partial_steps, ITER_PREFIX, latest_completed + 1,
        )

    if offset > 0:
        logger.info("Cumulative step offset: %d", offset)
    return offset
