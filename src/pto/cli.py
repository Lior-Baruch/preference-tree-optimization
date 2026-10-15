"""``pto`` command line: train | generate | evaluate | compare.

Heavy dependencies (torch, transformers, trl, the API SDKs) are imported inside the subcommand
handlers, so ``pto --help`` and ``pto train --dry-run`` stay fast and never touch a GPU or an API.

Examples::

    pto train --config configs/pto_k5.yaml
    pto train --config configs/quicktest.yaml --dry-run
    pto generate --config configs/pto_k5.yaml --adapter runs/<run>/iteration_10/adapter --out convs/
    pto evaluate --conversations runs/<run>/conversations/model_iter_10 \\
        --judge openai:gpt-4o-mini-2024-07-18 --out scores.csv
    pto compare base_scores.csv pto_scores.csv --out table.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__

logger = logging.getLogger("pto")

_DEFAULT_JUDGE = "openai:gpt-4o-mini-2024-07-18"


def _setup_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity > 0 else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if verbosity <= 1:
        # Third-party HTTP clients are chatty at INFO (one line per API request).
        for name in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)


def _load_cfg(path: str, overrides: dict[str, Any]) -> Any:
    from .config import load_config

    return load_config(path, overrides or None)


# ── Subcommand handlers ──────────────────────────────────────────────────────────


def _cmd_train(args: argparse.Namespace) -> int:
    overrides: dict[str, Any] = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.iterations is not None:
        overrides["loop.num_iterations"] = args.iterations
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    cfg = _load_cfg(args.config, overrides)

    if args.dry_run:
        from .config import config_to_yaml

        sys.stdout.write(config_to_yaml(cfg))
        sys.stdout.write(f"\n# run_dir: {cfg.run_dir}\n")
        return 0

    from .trainer import train

    run_dir = train(cfg)
    sys.stdout.write(f"{run_dir}\n")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args.config, {})
    from .trainer import generate

    adapter = None if args.adapter.lower() == "base" else args.adapter
    out = generate(
        cfg,
        adapter,
        args.out,
        num_conversations=args.num_conversations,
        seed=args.seed,
    )
    sys.stdout.write(f"{out}\n")
    return 0


def _parse_instruments(text: str | None) -> list[str] | None:
    # Names are validated (case-insensitively, with "Q1+Q2" -> Q1 and Q2) by evaluate_run, so
    # the CLI accepts exactly what the Python API accepts.
    if not text:
        return None
    return [s.strip() for s in text.split(",") if s.strip()]


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluation.judges import JudgeSpec
    from .evaluation.score import evaluate_run
    from .evaluation.stats import summarize, to_markdown

    judge = JudgeSpec.parse(args.judge)
    df = evaluate_run(
        args.conversations,
        judge,
        instruments=_parse_instruments(args.instruments),
        out_csv=args.out,
        concurrency=args.concurrency,
    )
    logger.info("Scores written to %s (%d rows)", args.out, len(df))
    sys.stdout.write(to_markdown(summarize(df)) + "\n")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    import pandas as pd

    from .evaluation.stats import paired_compare, to_markdown

    a = pd.read_csv(args.a)
    b = pd.read_csv(args.b)
    table = to_markdown(paired_compare(a, b, n_boot=args.n_boot, seed=args.seed))
    header = f"A = {args.a}\nB = {args.b}\ndelta = B - A, paired by persona_id\n\n"
    sys.stdout.write(header + table + "\n")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(header + table + "\n", encoding="utf-8")
        logger.info("Comparison table written to %s", args.out)
    return 0


# ── Parser ────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="more log output (-v: debug for pto, -vv: also HTTP clients)",
    )

    parser = argparse.ArgumentParser(
        prog="pto",
        description="Preference Tree Optimization: train, generate, evaluate, compare.",
    )
    # Separate dest: a subparser's default would otherwise reset a top-level -v.
    parser.add_argument(
        "-v", "--verbose", dest="verbose_global", action="count", default=0,
        help="more log output (also accepted after the subcommand)",
    )
    parser.add_argument("--version", action="version", version=f"pto {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p = sub.add_parser(
        "train", parents=[common], help="run (or resume) iterative PTO training",
        description="Run or resume iterative PTO training from a YAML config.",
    )
    p.add_argument("--config", required=True, help="YAML config (see configs/)")
    p.add_argument("--output-dir", help="override output_dir")
    p.add_argument("--iterations", type=int, help="override loop.num_iterations")
    p.add_argument("--run-name", help="override run_name")
    p.add_argument(
        "--dry-run", action="store_true",
        help="validate the config, print it and the run directory, then exit",
    )
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser(
        "generate", parents=[common],
        help="generate conversations with the base model or an adapter",
        description="Generate a conversation set (conversation_{i}.csv + manifest.json).",
    )
    p.add_argument("--config", required=True, help="YAML config (models + generation settings)")
    p.add_argument("--adapter", required=True, help="LoRA adapter path / Hub id, or 'base'")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--num-conversations", type=int, help="default: loop.conversations_per_iteration"
    )
    p.add_argument("--seed", type=int, help="persona-order and patient seed (default: config seed)")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser(
        "evaluate", parents=[common], help="score a conversation directory with an LLM judge",
        description="Score conversations with an LLM judge; resumable tidy CSV output.",
    )
    p.add_argument("--conversations", required=True, help="directory with conversation_{i}.csv")
    p.add_argument(
        "--judge", default=_DEFAULT_JUDGE,
        help=f"provider:model, e.g. {_DEFAULT_JUDGE} or anthropic:claude-haiku-4-5",
    )
    p.add_argument(
        "--instruments",
        help="comma-separated subset of Q1,Q2,WAI-SR,CSQ-8,MI-SAT,MITI,MICI,PCT (default: all)",
    )
    p.add_argument("--out", required=True, help="scores CSV (existing rows are reused)")
    p.add_argument("--concurrency", type=int, default=32, help="parallel judge calls")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser(
        "compare", parents=[common], help="paired comparison of two score CSVs",
        description="Compare two score CSVs paired by persona_id (delta = B - A).",
    )
    p.add_argument("a", help="scores CSV of run A (e.g. base model)")
    p.add_argument("b", help="scores CSV of run B (e.g. trained policy)")
    p.add_argument("--out", help="also write the Markdown table here")
    p.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples for the CI")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.set_defaults(func=_cmd_compare)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (console script ``pto``). Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose + args.verbose_global)

    from .clients import MissingAPIKeyError
    from .config import ConfigError

    try:
        return int(args.func(args))
    except ConfigError as e:
        sys.stderr.write(f"pto: {e}\n")
        return 2
    except (FileNotFoundError, ImportError, ValueError, MissingAPIKeyError) as e:
        logger.debug("Traceback", exc_info=True)
        sys.stderr.write(f"pto {args.command}: {type(e).__name__}: {e}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("pto: interrupted (progress so far is saved; rerun to resume)\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
