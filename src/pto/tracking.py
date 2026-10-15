"""Optional logging glue: TensorBoard / W&B lifecycle, checkpoint metadata, timing, EDA capture.

Everything here is best-effort bookkeeping around training. ``wandb`` and ``tensorboard`` are
imported lazily and only when ``logging.report_to`` asks for them.

- :func:`init_iteration_logging` / :func:`finish_iteration_logging`: one W&B run per iteration
  (each ``DPOTrainer`` restarts ``global_step`` at 0), grouped under the run name.
- :func:`setup_tensorboard_logging` / :func:`patch_trainer_tensorboard_callback`: pin the
  TensorBoard event directory to ``iteration_N/training/tb_logs``.
- :class:`CheckpointMetadataCallback`: writes ``experiment_metadata.json`` into every HF checkpoint.
- :func:`log_session` & co.: resume-proof, append-only per-iteration timing log.
- :class:`GenerationRecorder`: every branch candidate of an iteration (completion, score,
  sub-scores, look-ahead tail, chosen/rejected role) written once to ``eda/generations.jsonl``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _compat  # noqa: F401

# isort: split
from transformers import TrainerCallback

if TYPE_CHECKING:
    from .config import PTORunConfig

logger = logging.getLogger(__name__)

__all__ = [
    "SESSIONS_FILENAME",
    "CheckpointMetadataCallback",
    "GenerationRecorder",
    "build_wandb_ctx",
    "cumulative_seconds",
    "finish_iteration_logging",
    "finish_wandb",
    "init_iteration_logging",
    "log_session",
    "metadata_fields",
    "patch_trainer_tensorboard_callback",
    "read_sessions",
    "setup_tensorboard_logging",
]


# ── Trainer callback ──────────────────────────────────────────────────────────────


class CheckpointMetadataCallback(TrainerCallback):
    """Write ``experiment_metadata.json`` alongside each saved HF checkpoint.

    The file is extra payload inside ``checkpoint-N/`` and is ignored by HF/TRL resume.
    """

    def __init__(self, iteration: int, metadata: dict[str, Any]):
        self.iteration = iteration
        self.metadata = metadata

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(checkpoint_dir):
            return
        payload = {
            **self.metadata,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(checkpoint_dir, "experiment_metadata.json"), "w") as f:
            json.dump(payload, f, indent=2)


# ── W&B / TensorBoard lifecycle ───────────────────────────────────────────────────


def build_wandb_ctx(cfg: PTORunConfig) -> dict[str, Any] | None:
    """W&B context for :func:`init_iteration_logging`, or ``None`` when not reporting to W&B.

    The project defaults to ``pto`` (override with the ``WANDB_PROJECT`` environment variable);
    the run name groups the per-iteration runs.
    """
    if "wandb" not in cfg.logging.report_to:
        return None
    import wandb

    if wandb.run is not None:
        wandb.finish()
    ctx = {
        "run_id": cfg.run_name,
        "project": os.environ.get("WANDB_PROJECT", "pto"),
        "config": cfg.to_dict(),
    }
    logger.info("W&B: project=%s, group=%s", ctx["project"], ctx["run_id"])
    return ctx


def init_iteration_logging(
    report_to: list[str],
    iteration: int,
    cumulative_step_offset: int,
    wandb_ctx: dict[str, Any] | None,
) -> None:
    """Open this iteration's W&B run (one run per iteration, grouped under the run name).

    Called just before ``trainer.train()``. W&B's default ``train/global_step`` axis is kept; a
    custom cross-iteration step axis breaks the standard HF charts. ``cumulative_step_offset`` is
    therefore only recorded in the run config. TensorBoard needs nothing here: the HF Trainer
    creates its own ``SummaryWriter``.
    """
    if "wandb" in report_to and wandb_ctx is not None:
        import wandb

        if wandb.run is not None:
            wandb.finish()
        run_name = f"{wandb_ctx['run_id']}_iter{iteration}"
        wandb.init(
            project=wandb_ctx["project"],
            name=run_name,
            id=run_name,
            group=wandb_ctx["run_id"],
            resume="allow",
            config={
                **wandb_ctx["config"],
                "iteration": iteration,
                "cumulative_step_offset": cumulative_step_offset,
            },
        )
        logger.info("W&B run for iteration %d: %s", iteration, run_name)


def finish_iteration_logging(
    report_to: list[str], iteration: int, iter_level_metrics: dict[str, Any]
) -> None:
    """Log iteration-level summary metrics to W&B, then close this iteration's run."""
    if "wandb" in report_to:
        import wandb

        if wandb.run is not None:
            if iter_level_metrics:
                wandb.log(iter_level_metrics)
            wandb.finish()
            logger.info("W&B run for iteration %d finalized", iteration)


def finish_wandb(report_to: list[str]) -> None:
    """Close any W&B run still open at the end of training."""
    if "wandb" in report_to:
        import wandb

        if wandb.run is not None:
            wandb.finish()


def setup_tensorboard_logging(report_to: list[str], tensorboard_log_dir: str | None) -> None:
    """Point transformers' ``TensorBoardCallback`` at an explicit log directory.

    The callback reads ``TENSORBOARD_LOGGING_DIR`` at construction; an explicit directory also
    avoids invalid default paths on Windows.
    """
    if "tensorboard" not in report_to:
        os.environ.pop("TENSORBOARD_LOGGING_DIR", None)
        return
    if tensorboard_log_dir is None:
        raise ValueError("tensorboard_log_dir is required when reporting to tensorboard")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = tensorboard_log_dir
    logger.debug("TensorBoard log dir: %s", tensorboard_log_dir)


def patch_trainer_tensorboard_callback(trainer: Any, tensorboard_log_dir: str) -> None:
    """Force the trainer's ``TensorBoardCallback`` to write to ``tensorboard_log_dir``.

    Belt and braces for :func:`setup_tensorboard_logging`: some transformers versions resolve the
    directory from ``TrainingArguments`` instead of the environment variable.
    """
    for cb in trainer.callback_handler.callbacks:
        if cb.__class__.__name__ == "TensorBoardCallback":
            cb.logging_dir = tensorboard_log_dir
            if getattr(cb, "tb_writer", None) is not None:
                cb.tb_writer.close()
                cb.tb_writer = None
            return
    logger.warning("TensorBoardCallback not found on trainer; TensorBoard logs may be missing")


# ── Resume-proof per-iteration timing ─────────────────────────────────────────────
#
# ``iteration_metadata.json`` timings are per-process: an iteration that crashed and resumed
# reports only the last session (a resumed preference build that reloaded ``pairs.csv`` shows a
# few seconds). Each process that works on an iteration therefore appends ONE line to
# ``iteration_N/timing_sessions.jsonl``; cumulative cost is the sum over lines. Append-only means a
# crashed session's work stays recorded and two processes cannot lose each other's update.

#: One line per process that worked on the iteration. Append-only.
SESSIONS_FILENAME = "timing_sessions.jsonl"
#: Phases summed into ``total_s``; unknown keys are kept but never inflate a total.
PHASE_KEYS = ("generation_s", "pref_pair_s", "training_s")


def sessions_path(iter_dir: str) -> str:
    return os.path.join(iter_dir, SESSIONS_FILENAME)


def log_session(
    iter_dir: str,
    *,
    generation_s: float = 0.0,
    training_s: float = 0.0,
    pref_pair_s: float = 0.0,
    started_at: float | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Append this process's contribution to the iteration's timing log.

    Returns the record written, or ``{}`` if the write failed (timing never fails a run).
    """
    now = time.time()
    rec = {
        "generation_s": float(generation_s or 0.0),
        "pref_pair_s": float(pref_pair_s or 0.0),
        "training_s": float(training_s or 0.0),
        "wall_start": float(started_at) if started_at else None,
        "wall_end": now,
        "wall_span_s": (now - float(started_at)) if started_at else None,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "note": note,
    }
    try:
        os.makedirs(iter_dir, exist_ok=True)
        with open(sessions_path(iter_dir), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as e:  # telemetry must never break a run
        logger.warning("Could not append timing session log (%s)", e)
        return {}
    return rec


def read_sessions(iter_dir: str) -> list[dict[str, Any]]:
    """Every session recorded for this iteration; ``[]`` if absent. Skips unparseable lines."""
    fp = sessions_path(iter_dir)
    if not os.path.isfile(fp):
        return []
    out = []
    try:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line is not a reason to fail
    except OSError:
        return []
    return out


def cumulative_seconds(iter_dir: str) -> dict[str, float]:
    """Phase totals over every session, plus ``total_s`` and ``n_sessions``.

    ``n_sessions > 1`` means the iteration was resumed, i.e. its per-process timings are partial.
    """
    sessions = read_sessions(iter_dir)
    totals = {k: 0.0 for k in PHASE_KEYS}
    for s in sessions:
        for k in PHASE_KEYS:
            try:
                totals[k] += float(s.get(k) or 0.0)
            except (TypeError, ValueError):
                continue
    totals["total_s"] = sum(totals[k] for k in PHASE_KEYS)
    totals["n_sessions"] = float(len(sessions))
    return totals


def metadata_fields(iter_dir: str) -> dict[str, float]:
    """``cumulative_*`` fields to merge into ``iteration_metadata.json`` (``{}`` without a log)."""
    totals = cumulative_seconds(iter_dir)
    if not totals.get("n_sessions"):
        return {}
    return {
        "cumulative_generation_time_s": round(totals["generation_s"], 3),
        "cumulative_pref_pair_time_s": round(totals["pref_pair_s"], 3),
        "cumulative_training_time_s": round(totals["training_s"], 3),
        "cumulative_total_time_s": round(totals["total_s"], 3),
        "n_timing_sessions": int(totals["n_sessions"]),
    }


# ── Candidate-level generation records ────────────────────────────────────────────


def _to_jsonable(o: Any) -> Any:
    """``json.dump`` fallback for numpy scalars/arrays."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class GenerationRecorder:
    """In-memory, per-iteration buffer of branch records, flushed once to JSONL.

    DPO keeps only the (chosen, rejected) pair of each branch point; this keeps every candidate.
    One record per branch point, prefix stored once, candidates nested, look-ahead stored as the
    simulated tail only::

        {
          "method": "PTO", "iteration": 3, "phase": "tree" | "independent",
          "conversation_id": 17,      # index of the conversation in this iteration's persona order
          "branch_id": 24,            # greedy: trunk length at the branch; independent: turn index
          "prefix": "[THERAPIST]: ...\\n\\n[PATIENT]: ...",   # oracle transcript of the prefix
          "chosen_idx": 7 | None,
          "candidates": [
            {"idx": 0, "completion": "...", "score": 2.1 | None,
             "sub_scores": {"1": 2.0, "2": 2.2} | None,
             "role": "chosen" | "rejected" | "neither",
             "oracle": {"success": true, "retries": 0},
             "lookahead": {"k": 5, "realized_turns": 4, "ended_early": true, "tail": "..." | None}},
            ...
          ]
        }

    The oracle-scored text of a candidate is
    ``prefix + "\\n\\n[THERAPIST]: " + completion + (tail or "")``.

    Args:
        out_path: JSONL path written by :meth:`flush`.
        enabled: when False every method is a no-op.
        save_transcripts: when False the look-ahead ``tail`` is dropped at append time (scores
            and flags are kept); the main size lever for look-ahead runs.
    """

    def __init__(self, out_path: str, *, enabled: bool = True, save_transcripts: bool = True):
        self.out_path = out_path
        self.enabled = enabled
        self.save_transcripts = save_transcripts
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        """Buffer one branch record (cheap; safe inside the async scoring path)."""
        if not self.enabled:
            return
        if not self.save_transcripts:
            for cand in record.get("candidates", []):
                la = cand.get("lookahead")
                if isinstance(la, dict) and la.get("tail") is not None:
                    la["tail"] = None
        self.records.append(record)

    def flush(self) -> str | None:
        """Atomically write the buffer to ``out_path`` (tmp + ``os.replace``).

        Writes even an empty file so its presence reliably means "this build ran". Returns the
        path, or ``None`` when disabled.
        """
        if not self.enabled:
            return None
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        tmp = self.out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False, default=_to_jsonable))
                f.write("\n")
        os.replace(tmp, self.out_path)
        return self.out_path

    def aggregate(self) -> tuple[dict[str, float], list[float]]:
        """Per-iteration summary ``(scalars, scores)``; ``scores`` = every non-None score."""
        branches = self.records
        cands = [c for b in branches for c in b.get("candidates", [])]
        n = len(cands)
        scores = [float(c["score"]) for c in cands if c.get("score") is not None]
        scalars: dict[str, float] = {
            "eda/n_branches": float(len(branches)),
            "eda/n_candidates": float(n),
        }
        if scores:
            scalars["eda/mean_candidate_reward"] = float(np.mean(scores))
            scalars["eda/reward_std"] = float(np.std(scores))
        if n:
            scalars["eda/oracle_success_rate"] = float(
                np.mean([1.0 if (c.get("oracle") or {}).get("success") else 0.0 for c in cands])
            )
        la = [
            c["lookahead"]
            for c in cands
            if isinstance(c.get("lookahead"), dict) and (c["lookahead"].get("k") or 0) > 0
        ]
        if la:
            scalars["eda/lookahead_realized_turns_mean"] = float(
                np.mean([d.get("realized_turns", 0) for d in la])
            )
            scalars["eda/lookahead_ended_early_frac"] = float(
                np.mean([1.0 if d.get("ended_early") else 0.0 for d in la])
            )
        # A pair was emitted at a branch point iff it has a "rejected" candidate.
        if branches:
            bp = len(branches)
            pp = sum(
                1
                for b in branches
                if any(c.get("role") == "rejected" for c in b.get("candidates", []))
            )
            scalars["pto/branch_points"] = float(bp)
            scalars["pto/pref_pair_count"] = float(pp)
            scalars["pto/tau_filter_rate"] = float(1.0 - pp / bp)
        return scalars, scores
