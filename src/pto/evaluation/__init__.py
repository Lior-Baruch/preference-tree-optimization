"""Judge-based evaluation of saved conversations and paired statistics between runs.

Typical use::

    from pto.evaluation import JudgeSpec, evaluate_run, paired_compare, to_markdown

    judge = JudgeSpec.parse("openai:gpt-4o-mini-2024-07-18")
    base = evaluate_run("runs/<run>/conversations/model_iter_0", judge, out_csv="base.csv")
    final = evaluate_run("runs/<run>/conversations/model_iter_10", judge, out_csv="final.csv")
    print(to_markdown(paired_compare(base, final)))
"""

from .judges import JudgeSpec
from .score import INSTRUMENTS, evaluate_run, evaluate_run_async
from .stats import paired_compare, summarize, to_markdown

__all__ = [
    "INSTRUMENTS",
    "JudgeSpec",
    "evaluate_run",
    "evaluate_run_async",
    "paired_compare",
    "summarize",
    "to_markdown",
]
