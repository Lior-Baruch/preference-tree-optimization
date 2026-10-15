"""PTO: Preference Tree Optimization for multi-turn therapist dialogue policies.

Each iteration the current policy talks to an LLM-simulated patient; at every therapist turn it
samples several candidate replies, optionally rolls each one forward K turns (look-ahead), has an
LLM oracle score the result with a clinical questionnaire, and turns the best/worst candidates into
DPO preference pairs.

The package import is intentionally light (no torch/trl/transformers); heavy dependencies are
loaded by the submodules that need them. Entry points: :func:`pto.trainer.train`,
:func:`pto.trainer.generate`, :func:`pto.evaluation.score.evaluate_run` and the ``pto`` CLI.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
