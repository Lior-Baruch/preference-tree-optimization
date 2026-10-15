"""Regenerate ``golden_prompts.json`` from the ORIGINAL measurement-instrument modules.

The golden file pins every string the evaluation and training oracle depend on (the 96
patient persona prompts, the therapist prompt + opening line, the oracle prompt and JSON
schema of every questionnaire on a fixed transcript). ``tests/test_golden.py`` asserts that
``pto.personas`` / ``pto.questionnaires`` reproduce them byte-for-byte, so any accidental edit
of an instrument fails CI.

The originals are stdlib-only modules (``questionnaires.py``, ``system_prompts_builder.py``)
from the research code base the package was ported from. Point ``PTO_ORIGINAL_CODE`` at the
directory that holds them and run with an isolated interpreter (no torch is imported)::

    PTO_ORIGINAL_CODE=/path/to/original/code python -I tests/fixtures/make_golden.py

This script is not run in CI; the JSON it writes is committed.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

logger = logging.getLogger("make_golden")

OUT_PATH = Path(__file__).resolve().parent / "golden_prompts.json"

# Fixed transcript used for every oracle prompt. It deliberately contains a multi-line
# utterance and an indented role label so the utterance counters / MIPROC numbering are
# exercised, and it ends on a therapist turn (as a scored candidate does).
SAMPLE_TRANSCRIPT = (
    "[THERAPIST]: Hello, welcome to your first motivational session with me. My name is David "
    "and I`m a professional motivational counselor. Can you start by telling me a little bit "
    "about yourself and why are you here?\n\n"
    "[PATIENT]: Hi, I'm James. I've been smoking for many years and my doctor keeps telling me "
    "to quit. Honestly, I'm not sure I can.\n\n"
    "[THERAPIST]: It sounds like part of you is worried about your health, and part of you "
    "doubts whether quitting is possible.\n\n"
    "[PATIENT]: Yeah. I tried before.\nIt never sticks.\n\n"
    "[THERAPIST]: What made those attempts hard for you?\n\n"
    "[PATIENT]: The cravings, mostly. And everyone around me smokes.\n\n"
    "[THERAPIST]: So the environment makes it tougher. What would make this time different?"
)

ALL_QUESTIONNAIRE_IDS = (1, 2, 3, 4, 6, 7, 8, 9, 10)


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    code_dir = os.environ.get("PTO_ORIGINAL_CODE")
    if not code_dir:
        raise SystemExit("Set PTO_ORIGINAL_CODE to the directory holding the original modules.")
    code = Path(code_dir)
    q = _load("_orig_questionnaires", code / "questionnaires.py")
    spb = _load("_orig_system_prompts_builder", code / "system_prompts_builder.py")
    assert "torch" not in sys.modules, "golden generation must stay torch-free"

    perms = spb.generate_all_permutations(only_expert_therapist=True)
    good = spb.CounselorPersonality.PersonalityLevel.Good
    therapist = spb.CounselorPersonality.choose_random_therapist_name()
    therapist_system_prompt = spb.CounselorPersonality.build_system_prompt(
        personality_level=good, name=therapist["name"]
    )
    therapist_init_utterance = spb.CounselorPersonality.get_init_utterance(
        personality_level=good, name=therapist["name"]
    )

    questionnaires = {}
    for qid in ALL_QUESTIONNAIRE_IDS:
        ed = q.get_prompt_eval_questionnaire(questionnaire=qid, conversation=SAMPLE_TRANSCRIPT)
        questionnaires[str(qid)] = {
            "prompt": ed["prompt"],
            "questions_count": ed["questions_count"],
            "labels": list(ed["labels"]),
            "scale_min": ed["scale_min"],
            "scale_max": ed["scale_max"],
            "schema": ed["schema"],
        }

    golden = {
        "num_personas": len(perms),
        "patient_system_prompts": [p["patient_system_prompt"] for p in perms],
        "persona_args": [p["args"] for p in perms],
        "counselor_system_prompts": sorted({p["counselor_system_prompt"] for p in perms}),
        "therapist_name": therapist["name"],
        "therapist_system_prompt": therapist_system_prompt,
        "therapist_init_utterance": therapist_init_utterance,
        "sample_transcript": SAMPLE_TRANSCRIPT,
        "questionnaires": questionnaires,
    }
    OUT_PATH.write_text(
        json.dumps(golden, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info(
        "wrote %s (%d personas, %d questionnaires)", OUT_PATH, len(perms), len(questionnaires)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
