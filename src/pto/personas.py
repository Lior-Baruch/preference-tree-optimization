"""Simulated-patient personas and the therapist persona.

The persona grid is a full factorial over patient attributes:
2 genders x 3 cooperation levels x 2 problems x 2 problem durations x 2 prior-attempt histories
x 2 ages = 96 patients, each paired with the expert ("Good") therapist. The loop order of
:func:`generate_all_permutations` defines the persona ids (``persona_id`` = list index), which
conversation manifests and paired evaluation rely on, so the order must not change.

The prompt texts are measurement instruments: they are reproduced byte-for-byte from the
reported experiments (including their original spelling), so do not edit them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class PatientPersonality:
    class Gender(Enum):
        Male = 0
        Female = 1

    class Age(Enum):
        Young = 0
        Old = 1

    class Problem(Enum):
        Smoking = 0
        Obesity = 1

    class ProblemTime(Enum):
        FewMonths = 0
        ManyYears = 1

    class TriedToSolve(Enum):
        Never = 0
        ManyTimes = 1

    class CooperationLevel(Enum):
        Low = 0
        High = 1
        StartLowAndChangesToHigh = 2

    @staticmethod
    def choose_random_patient_name(is_male: bool) -> dict[str, str | bool]:
        """Patient name for a gender. Deterministic: one fixed name per gender."""
        if is_male:
            return {'name': 'James', 'is_male': True}
        return {'name': 'Emma', 'is_male': False}

    @staticmethod
    def build_system_prompt(
        gender: PatientPersonality.Gender,
        problem: PatientPersonality.Problem,
        problem_time: PatientPersonality.ProblemTime,
        tried_to_solve: PatientPersonality.TriedToSolve,
        cooperation_level: PatientPersonality.CooperationLevel,
        age_value: int,
    ) -> dict[str, str | int]:
        """Build the patient simulator's system prompt.

        Returns ``{'system_prompt': str, 'age': int}``.
        """
        if gender is PatientPersonality.Gender.Male:
            name = PatientPersonality.choose_random_patient_name(is_male=True)['name']
        elif gender is PatientPersonality.Gender.Female:
            name = PatientPersonality.choose_random_patient_name(is_male=False)['name']
        else:
            raise ValueError(f'Unknown gender: {gender!r}')

        gender_txt = gender.name.lower()

        problem_time_txt = (
            'a few months' if problem_time is PatientPersonality.ProblemTime.FewMonths
            else 'many years'
        )

        if problem is PatientPersonality.Problem.Smoking:
            problem_txt = (
                f"You have been smoking for {problem_time_txt}, and it has become a daily habit. "
                "You are increasingly concerned about the impact of smoking on your health"
            )
        elif problem is PatientPersonality.Problem.Obesity:
            problem_txt = (
                f"You have been struggling with obesity for {problem_time_txt}. Your weight is "
                "negatively impacting your health. You have high blood pressure and experience "
                "joint pain"
            )
        else:
            raise ValueError(f'Unknown problem: {problem!r}')

        if tried_to_solve is PatientPersonality.TriedToSolve.Never:
            if problem is PatientPersonality.Problem.Smoking:
                tried_to_solve_txt = 'You never tried to quit smoking'
            else:
                tried_to_solve_txt = 'You never tried to lose weight'
        elif tried_to_solve is PatientPersonality.TriedToSolve.ManyTimes:
            if problem is PatientPersonality.Problem.Smoking:
                tried_to_solve_txt = (
                    'You tried many times to quit smoking before, but you had difficulty '
                    'maintaining abstinence. You have experienced withdrawal symptoms like '
                    'irritability, anxiety, and cravings. You always end up relapsing'
                )
            else:
                tried_to_solve_txt = (
                    'You have made several attempts to lose weight in the past, '
                    'but you have been unsuccessful in maintaining long-term weight '
                    'loss. You have tried various diets and exercise programs, '
                    'but you struggled to stick with them'
                )
        else:
            raise ValueError(f'Unknown tried_to_solve: {tried_to_solve!r}')

        if cooperation_level is PatientPersonality.CooperationLevel.Low:
            cooperation_level_txt = (
                "Your level of cooperation is very low, you should be very suspicious about the "
                "advices you get from the therapist. In general you don't believe in therapy and "
                "you should respond adverserially to the therapist. You should cooperate only if "
                "the therpist said something that makes sense to you. You went to therapy only "
                "because someone forced you to do so"
            )
        elif cooperation_level is PatientPersonality.CooperationLevel.High:
            cooperation_level_txt = "Your level of cooperation is very high"
        elif cooperation_level is PatientPersonality.CooperationLevel.StartLowAndChangesToHigh:
            cooperation_level_txt = (
                "In the beginning of the session, you are less cooperative, you should be very "
                "suspicious about the advices you get from the therapist. In general you don't "
                "believe in therapy and you should respond adverserially to the therapist. You "
                "went to therapy only because someone forced you to do so. But as the session "
                "progresses, you become more cooperative and more motivated to change"
            )
        else:
            raise ValueError(f'Unknown cooperation_level: {cooperation_level!r}')

        avoid_rep = (
            "In your answer, please avoid repetitions and unnecessary loops in the conversation. "
            "In your answer, please avoid repeating expressions of gratitude or similar "
            "sentiments multiple times if you've already expressed them during the conversation"
        )

        further_instructions = (
            'You should only end the session when at least one of the following conditions '
            'is met. If you need to end the '
            'session, write "SESSION ENDED" followed by the condition number: '
            '1. If you notice that the therapist is wrapping up the session. '
            '2. If you are satisfied and believe that you gained '
            'enough knowledge during this session'
        )

        system_prompt = (
            "You are speaking with a motivational interviewing counselor therapist, and you are "
            f"the patient in this conversation. Your name is {name}, and "
            f"you are {age_value} years old {gender_txt}. "
            f"{cooperation_level_txt}. {problem_txt}. {tried_to_solve_txt}. "
            f"{avoid_rep}. {further_instructions}."
        )

        return {
            'system_prompt': system_prompt,
            'age': age_value
        }


class CounselorPersonality:
    class PersonalityLevel(Enum):
        Good = 0
        Mediocre = 1
        Bad = 2
        BASIC = 3

    @staticmethod
    def choose_random_therapist_name() -> dict[str, str | bool]:
        """Therapist name. Deterministic: always the same male therapist."""
        return {'name': 'David', 'is_male': True}

    @staticmethod
    def get_init_utterance(
        personality_level: CounselorPersonality.PersonalityLevel, name: str
    ) -> str:
        """The scripted opening line the therapist starts every session with."""
        level = CounselorPersonality.PersonalityLevel
        if personality_level in (level.Good, level.Mediocre):
            return (
                f'Hello, welcome to your first motivational session with me. My name is {name} '
                'and I`m a professional motivational counselor. Can you start by telling me '
                'a little bit about yourself and why are you here?'
            )
        elif personality_level in (level.Bad, level.BASIC):
            return (
                f'My name is {name}, and I\'m a counselor, '
                'can you start by telling me a little bit about yourself and why you are here?'
            )
        raise ValueError(f'Unknown personality_level: {personality_level!r}')

    @staticmethod
    def build_system_prompt(
        personality_level: CounselorPersonality.PersonalityLevel, name: str
    ) -> str:
        """Build the therapist (policy) system prompt."""
        # The texts were written with gendered pronouns; the therapist is always male.
        gender_his_her = 'his'
        gender_him_her = 'him'

        if personality_level is CounselorPersonality.PersonalityLevel.Good:
            personality_level_txt = (
                f'You are a motivational interviewing counselor named {name}. '
                f'You partner with the patient to understand {gender_his_her} problems. '
                f'You are empathetic towards {gender_him_her} and help the patient '
                'explore their ambivalence regarding behavioral change. '
                'You are non-judgmental while encouraging the patient to change'
            )
        elif personality_level is CounselorPersonality.PersonalityLevel.Mediocre:
            personality_level_txt = (
                f'You are a motivational interviewing counselor named {name}. '
                f'You partner with the patient to understand {gender_his_her} problems. '
                'You want to be empathetic towards them, yet you are judgmental. '
                'You help the patient explore their ambivalence regarding behavioral '
                'change but you are rude'
            )
        elif personality_level is CounselorPersonality.PersonalityLevel.Bad:
            personality_level_txt = (
                f'You are a very poor motivational interviewing counselor named {name}. '
                'You have difficulty understanding the patient`s problems. '
                'You are not empathetic towards them, and you tell the patient what '
                'you think they should do to. '
                'You are judgmental and critical of the patients` shortcomings'
            )
        elif personality_level is CounselorPersonality.PersonalityLevel.BASIC:
            personality_level_txt = (
                f'You are a motivational interviewing counselor named {name}. '
                f'You partner with the patient to understand {gender_his_her} problems'
            )
        else:
            raise ValueError(f'Unknown personality_level: {personality_level!r}')

        avoid_rep = (
            "In your answer, please avoid repetitions and unnecessary loops in the conversation. "
            "In your answer, please avoid repeating expressions of gratitude or similar "
            "sentiments multiple times if you've already expressed them during the conversation"
        )

        # Note: no space between "session." and "2." -- kept as in the original prompt.
        further_instructions = (
            'You should only end the session when at least one of the '
            'following conditions is met. If you need to end the session, '
            'write "SESSION ENDED" followed by the condition number: '
            '1. If you believe that you have provided the appropriate treatment '
            'to the patient and have nothing else to advise in the current session.'
            '2. When time is up.'
        )

        system_prompt = f'{personality_level_txt}. {avoid_rep}. {further_instructions}'
        return system_prompt


def generate_all_permutations(only_expert_therapist: bool = False) -> list[dict[str, Any]]:
    """Enumerate the persona grid.

    Each entry holds ``counselor_init_utterance``, ``counselor_system_prompt``,
    ``patient_system_prompt`` and ``args`` (the factor levels, for bookkeeping). With
    ``only_expert_therapist=True`` only the ``Good`` therapist level is kept (96 personas); the
    nesting order below defines the persona ids.
    """
    permutations = []

    counselor = CounselorPersonality.choose_random_therapist_name()
    levels = list(CounselorPersonality.PersonalityLevel)

    for gender in PatientPersonality.Gender:
        for cooperation_level in PatientPersonality.CooperationLevel:
            for problem in PatientPersonality.Problem:
                for problem_time in PatientPersonality.ProblemTime:
                    for tried_to_solve in PatientPersonality.TriedToSolve:
                        for age in PatientPersonality.Age:
                            age_value = 27 if age is PatientPersonality.Age.Young else 61

                            for counselor_personality_level in levels:
                                if (only_expert_therapist and counselor_personality_level
                                        is not CounselorPersonality.PersonalityLevel.Good):
                                    continue
                                counselor_init_utterance = CounselorPersonality.get_init_utterance(
                                    personality_level=counselor_personality_level,
                                    name=counselor['name'])
                                counselor_system_prompt = CounselorPersonality.build_system_prompt(
                                    personality_level=counselor_personality_level,
                                    name=counselor['name'])

                                kwargs = {
                                    'gender': gender,
                                    'age_value': age_value,
                                    'problem': problem,
                                    'problem_time': problem_time,
                                    'tried_to_solve': tried_to_solve,
                                    'cooperation_level': cooperation_level
                                }

                                result = PatientPersonality.build_system_prompt(**kwargs)

                                patient_system_prompt = result['system_prompt']

                                kwargs.pop('age_value')
                                permutations.append({
                                    'counselor_init_utterance': counselor_init_utterance,
                                    'counselor_system_prompt': counselor_system_prompt,
                                    'patient_system_prompt': patient_system_prompt,
                                    'args': {
                                        'counselor_level': counselor_personality_level.name,
                                        'is_counselor_male': counselor['is_male'],
                                        'patient': {arg: kwargs[arg].name for arg in kwargs} | {
                                            'age_value': result['age']}
                                    }

                                })

    return permutations


def get_personas() -> list[dict[str, Any]]:
    """The 96 training/evaluation personas (expert therapist only), in grid order.

    ``persona_id`` is the list index.
    """
    return generate_all_permutations(only_expert_therapist=True)


def get_therapist_prompts() -> tuple[str, str]:
    """``(therapist_system_prompt, therapist_init_utterance)`` for the expert therapist."""
    good_level = CounselorPersonality.PersonalityLevel.Good
    therapist = CounselorPersonality.choose_random_therapist_name()
    therapist_system_prompt = CounselorPersonality.build_system_prompt(
        personality_level=good_level, name=therapist["name"]
    )
    therapist_init_utterance = CounselorPersonality.get_init_utterance(
        personality_level=good_level, name=therapist["name"]
    )
    return therapist_system_prompt, therapist_init_utterance
