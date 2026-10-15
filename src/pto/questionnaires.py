# ruff: noqa: E501
# (Questionnaire texts are measurement instruments and must stay byte-identical, so their long
# lines are not re-wrapped.)
"""Questionnaire definitions used to score therapist-patient conversations with an LLM judge.

Every questionnaire is answered as strict JSON (OpenAI ``response_format`` JSON schema), so no
free-text parsing is needed.

Supported questionnaires:
- Q1 + Q2: 5 + 17 item therapist-performance questionnaires (IDs 1, 2); the PTO training reward
- WAI-SR: Working Alliance Inventory - Short Revised (ID 3)
- CSQ-8: Client Satisfaction Questionnaire (ID 4)
- MI-SAT: MI Intervention Satisfaction (ID 6)
- MITI: MI Treatment Integrity - Globals + Behaviors (ID 7)
- PCT: Patient Change Talk - Globals + patient-utterance counts (ID 8)
- MICI: MI-Inconsistent therapist behaviors - Global + counts (ID 9)
- MIPROC: MI process coder - ONE code per utterance, both speakers (ID 10)

Only the flat ``scores``-array questionnaires (1, 2, 3, 4, 6) can serve as a training oracle; the
nested ones (7-10) are evaluation-only.

Prompt layout matters: :func:`get_prompt_eval_questionnaire` puts the instructions and rubric
FIRST and the transcript LAST. The transcript-independent prefix is what makes provider-side
prompt caching effective, and the layout must not change for scores to stay comparable.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any


class QuestionnaireID(Enum):
    """Enumeration of questionnaire IDs."""
    Q1 = 1
    Q2 = 2
    WAI_SR = 3
    CSQ8 = 4
    MI_SAT = 6
    MITI = 7
    PCT = 8   # Patient Change Talk (patient-perspective; change/sustain talk + readiness)
    MICI = 9  # MI-Inconsistent therapist behaviors (negative-valence; higher = worse)
    MIPROC = 10  # MI process coder: ONE MITI/MISC-style code per utterance, BOTH speakers


class Questionnaire:
    """Questionnaire definition with JSON schema support."""

    def __init__(
        self,
        questionnaire_id: int,
        questions_count: int,
        questionnaire_prompt: str,
        labels: list[str],
        scale_min: int,
        scale_max: int,
    ):
        self.questionnaire_id = questionnaire_id
        self.questions_count = questions_count
        self.questionnaire_prompt = questionnaire_prompt
        self.labels = labels
        self.scale_min = scale_min
        self.scale_max = scale_max


# =============================================================================
# JSON SCHEMA BUILDERS
# =============================================================================

def make_eval_schema(
    questionnaire_id: QuestionnaireID | int,
    n_questions: int,
    scale_min: int = 1,
    scale_max: int = 5,
) -> dict:
    """
    JSON Schema for structured questionnaire evaluation output.

    Args:
        questionnaire_id: Unique ID for the questionnaire (QuestionnaireID enum or int)
        n_questions: Number of questions/items
        scale_min: Minimum score value (inclusive)
        scale_max: Maximum score value (inclusive)

    Returns:
        JSON schema dict for OpenAI response_format
    """
    qid = questionnaire_id.value if isinstance(questionnaire_id, QuestionnaireID) else questionnaire_id
    return {
        "type": "object",
        "properties": {
            "questionnaire_id": {"type": "integer", "enum": [int(qid)]},
            "scores": {
                "type": "array",
                "items": {"type": "integer", "minimum": scale_min, "maximum": scale_max},
                "minItems": int(n_questions),
                "maxItems": int(n_questions),
            },
        },
        "required": ["questionnaire_id", "scores"],
        "additionalProperties": False,
    }


def make_miti_schema(questionnaire_id: QuestionnaireID | int = QuestionnaireID.MITI) -> dict:
    """
    JSON Schema for MITI 4.2 evaluation (4 globals + 7 behavior counts).

    Returns:
        JSON schema dict for OpenAI response_format
    """
    qid = questionnaire_id.value if isinstance(questionnaire_id, QuestionnaireID) else questionnaire_id
    return {
        "type": "object",
        "properties": {
            "questionnaire_id": {"type": "integer", "enum": [int(qid)]},
            "globals": {
                "type": "object",
                "properties": {
                    "MITI1_CultivatingChangeTalk": {"type": "integer", "minimum": 1, "maximum": 5},
                    "MITI2_SofteningSustainTalk": {"type": "integer", "minimum": 1, "maximum": 5},
                    "MITI3_Partnership": {"type": "integer", "minimum": 1, "maximum": 5},
                    "MITI4_Empathy": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": [
                    "MITI1_CultivatingChangeTalk",
                    "MITI2_SofteningSustainTalk",
                    "MITI3_Partnership",
                    "MITI4_Empathy"
                ],
                "additionalProperties": False,
            },
            "behaviors": {
                "type": "object",
                "properties": {
                    "MITI_B1_GI": {"type": "integer", "minimum": 0},
                    "MITI_B2_Persuade": {"type": "integer", "minimum": 0},
                    "MITI_B3_Q": {"type": "integer", "minimum": 0},
                    "MITI_B4_SR": {"type": "integer", "minimum": 0},
                    "MITI_B5_CR": {"type": "integer", "minimum": 0},
                    "MITI_B6_AF": {"type": "integer", "minimum": 0},
                    "MITI_B7_Seek": {"type": "integer", "minimum": 0},
                },
                "required": [
                    "MITI_B1_GI",
                    "MITI_B2_Persuade",
                    "MITI_B3_Q",
                    "MITI_B4_SR",
                    "MITI_B5_CR",
                    "MITI_B6_AF",
                    "MITI_B7_Seek"
                ],
                "additionalProperties": False,
            },
        },
        "required": ["questionnaire_id", "globals", "behaviors"],
        "additionalProperties": False,
    }


def make_pct_schema(questionnaire_id: QuestionnaireID | int = QuestionnaireID.PCT) -> dict:
    """JSON Schema for PCT (Patient Change Talk): 3 globals (1-5) + 3 patient-utterance counts."""
    qid = questionnaire_id.value if isinstance(questionnaire_id, QuestionnaireID) else questionnaire_id
    return {
        "type": "object",
        "properties": {
            "questionnaire_id": {"type": "integer", "enum": [int(qid)]},
            "globals": {
                "type": "object",
                "properties": {
                    "PCT_Importance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "PCT_Confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                    "PCT_Readiness": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["PCT_Importance", "PCT_Confidence", "PCT_Readiness"],
                "additionalProperties": False,
            },
            "behaviors": {
                "type": "object",
                "properties": {
                    "PCT_ChangeTalk": {"type": "integer", "minimum": 0},
                    "PCT_SustainTalk": {"type": "integer", "minimum": 0},
                    "PCT_Neutral": {"type": "integer", "minimum": 0},
                },
                "required": ["PCT_ChangeTalk", "PCT_SustainTalk", "PCT_Neutral"],
                "additionalProperties": False,
            },
        },
        "required": ["questionnaire_id", "globals", "behaviors"],
        "additionalProperties": False,
    }


def make_mici_schema(questionnaire_id: QuestionnaireID | int = QuestionnaireID.MICI) -> dict:
    """JSON Schema for MICI (MI-Inconsistent behaviors): 1 global (1-5) + 6 therapist-utterance counts."""
    qid = questionnaire_id.value if isinstance(questionnaire_id, QuestionnaireID) else questionnaire_id
    return {
        "type": "object",
        "properties": {
            "questionnaire_id": {"type": "integer", "enum": [int(qid)]},
            "globals": {
                "type": "object",
                "properties": {
                    "MICI_Severity": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["MICI_Severity"],
                "additionalProperties": False,
            },
            "behaviors": {
                "type": "object",
                "properties": {
                    "MICI_Confront": {"type": "integer", "minimum": 0},
                    "MICI_AdviseNoPermission": {"type": "integer", "minimum": 0},
                    "MICI_Warn": {"type": "integer", "minimum": 0},
                    "MICI_Direct": {"type": "integer", "minimum": 0},
                    "MICI_Judge": {"type": "integer", "minimum": 0},
                    "MICI_OverPraise": {"type": "integer", "minimum": 0},
                },
                "required": [
                    "MICI_Confront", "MICI_AdviseNoPermission", "MICI_Warn",
                    "MICI_Direct", "MICI_Judge", "MICI_OverPraise",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["questionnaire_id", "globals", "behaviors"],
        "additionalProperties": False,
    }


# MIPROC therapist / patient code vocabularies (MITI 4.2.1 / MISC grounded). The order here is the
# column order of the per-code counts in the evaluation output.
MIPROC_TH_CODES = ["OQ", "CQ", "SR", "CR", "AF", "PRA", "GI", "PERS", "SEEK", "CONF", "OTH"]
MIPROC_PT_CODES = ["CT", "ST", "NEU"]


def make_miproc_schema(
    questionnaire_id: QuestionnaireID | int = QuestionnaireID.MIPROC,
    n_therapist: int = 0,
    n_patient: int = 0,
) -> dict:
    """JSON Schema for MIPROC: one code per therapist utterance + one per patient utterance.

    Both arrays are length-pinned (``minItems == maxItems`` == that speaker's utterance count) so
    the coder returns exactly one code per utterance, in transcript order — the same guarantee
    ``make_eval_schema`` gives the Likert rubrics. The Anthropic judge path
    (``evaluation.judges._strip_unsupported_constraints``) restates the pinned length in
    ``description``; ``parse_json_response(expected_counts=...)`` re-validates the lengths
    client-side either way. Tolerates ``n_therapist == n_patient == 0`` (an empty transcript).
    """
    qid = questionnaire_id.value if isinstance(questionnaire_id, QuestionnaireID) else questionnaire_id
    return {
        "type": "object",
        "properties": {
            "questionnaire_id": {"type": "integer", "enum": [int(qid)]},
            "therapist_codes": {
                "type": "array",
                "description": "One code per [THERAPIST #i] utterance, in transcript order.",
                "items": {"type": "string", "enum": list(MIPROC_TH_CODES)},
                "minItems": int(n_therapist),
                "maxItems": int(n_therapist),
            },
            "patient_codes": {
                "type": "array",
                "description": "One code per [PATIENT #i] utterance, in transcript order.",
                "items": {"type": "string", "enum": list(MIPROC_PT_CODES)},
                "minItems": int(n_patient),
                "maxItems": int(n_patient),
            },
        },
        "required": ["questionnaire_id", "therapist_codes", "patient_codes"],
        "additionalProperties": False,
    }


# =============================================================================
# Q1 + Q2 (Original Questionnaires)
# =============================================================================

Q1_LABELS = [f"Q1_{i}" for i in range(1, 6)]  # Q1_1 to Q1_5
Q2_LABELS = [f"Q2_{i}" for i in range(1, 18)]  # Q2_1 to Q2_17


def get_questionnaire_1() -> Questionnaire:
    """Questionnaire 1: 5 items, 1-5 Likert."""
    prompt = '''1. Your overall satisfaction with the chat?
This question aims to capture the general sense of how pleased or content a person was with their conversation with the therapist. Factors that could influence this rating might include the therapist's responsiveness, clarity of the responses, understanding of questions, and the general feeling that the conversation was useful or enjoyable.
Good response example: The therapist provides relevant and helpful responses to the patient's inquiries in a timely manner, maintaining a courteous and respectful tone throughout the conversation.
Bad response example: The therapist misunderstands the patient's questions frequently, provides irrelevant information, or responds with a significant delay.
2. Your overall satisfaction with the content of the chat?
This question is related to the actual substance of the therapist's responses. It asks about the quality, relevance, and helpfulness of the information provided by the therapist.
Good response example: The therapist provides accurate, detailed, and pertinent answers, supported by evidence or thoughtful analysis where appropriate.
Bad response example: The therapist provides vague, incorrect, or unhelpful answers, or frequently veers off topic.
3. To which extent do you feel the chat facilitated motivation?
This question measures the ability of the therapist to inspire, encourage, or stimulate the patient's interest or action towards a certain topic or goal. It's about whether the conversation made the patient feel more motivated.
Good response example: The therapist suggests practical steps to achieve the patient's goal, provides uplifting messages, or encourages perseverance, thus leading to an increase in the patient's motivation.
Bad response example: The therapist's responses are largely negative, pessimistic, or uninspiring, which could potentially diminish the patient's motivation.
4. Did you learn anything?
This question directly asks if the patient gained new knowledge or insights from the conversation with the therapist. It's about whether the therapist was able to teach something to the patient.
Good response example: The therapist offers well-informed and insightful answers, providing useful and new information that the patient wasn't aware of before the chat.
Bad response example: The therapist's responses are superficial or incorrect, and don't contribute to the patient's understanding of the topic in question.
5. To what extent was this learning relevant to your everyday life?
This question is about the applicability or usefulness of the knowledge or insight gained from the conversation. It's about how much the patient can take from the conversation and use in their day-to-day life.
Good response example: The therapist provides advice or information that directly relates to challenges or tasks that the patient faces regularly, thus leading to a high level of applicability to their everyday life.
Bad response example: The therapist's information or advice is largely theoretical, too complex, or irrelevant to the patient's life and circumstances, leading to low applicability.'''

    return Questionnaire(
        questionnaire_id=1,
        questions_count=5,
        questionnaire_prompt=prompt,
        labels=Q1_LABELS,
        scale_min=1,
        scale_max=5,
    )


def get_questionnaire_2() -> Questionnaire:
    """Questionnaire 2: 17 items, 1-5 Likert."""

    prompt = '''1. The therapist gave me a sense of who he was
This question seeks to understand if the therapist provided a sense of identity or persona.
Good response example: The therapist maintains a consistent vocabulary, style of writing, or approach that allows patients to understand its characteristics or personality.
Bad response example: The therapist's responses vary widely in vocabulary, writing or approach, making it difficult for patients to form a consistent understanding of the therapist's 'persona'.
2. The therapist revealed what he was thinking
This question asks if the therapist explained its thought process or reasoning. Providing such transparency can improve patient trust and understanding.
Good response example: The therapist often explains why it asks certain questions or provides specific responses.
Bad response example: The therapist gives answers or asks questions without providing any context or reasoning.
3. The therapist shared his feelings with me
This question is about whether the therapist expressed emotions in its responses. This can humanize the therapist and make interactions more relatable.
Good response example: The therapist uses phrases such as "I'm happy to help" or "I'm sorry for the inconvenience".
Bad response example: The therapist provides purely factual responses, without any emotional language.
4. The therapist seemed to know how I was feeling
This question measures the therapist's emotional intelligence, or its ability to understand and respond to the patient's emotions.
Good response example: When a patient expresses frustration or excitement, the therapist acknowledges and responds to these emotions appropriately.
Bad response example: The therapist doesn't acknowledge or respond to the patient's emotions or responds inappropriately.
5. The therapist seemed to understand me
This question asks whether the therapist understood the patient's inquiries and provided relevant responses.
Good response example: The therapist accurately interprets the patient's questions and provides relevant, accurate answers.
Bad response example: The therapist frequently misinterprets questions or provides irrelevant responses.
6. The therapist put hisself in my shoes
This question gauges the therapist's ability to empathize with the patient, considering their perspective and emotions.
Good response example: The therapist acknowledges the patient's feelings, offers understanding responses, and provides appropriate advice or solutions.
Bad response example: The therapist seems indifferent or dismissive of the patient's feelings or perspective.
7. The therapist seemed to be comfortable talking with me
This question is about whether the therapist provided smooth, natural responses, creating an impression of being at ease in the conversation.
Good response example: The therapist provides timely, coherent responses that flow naturally in the conversation.
Bad response example: The therapist's responses are delayed, disjointed, or awkwardly phrased.
8. The therapist seemed relaxed and secure when talking with me
This question asks whether the therapist conveyed a sense of confidence and assurance in its interactions.
Good response example: The therapist maintains consistent and clear communication and handles patient's questions confidently.
Bad response example: The therapist often provides unclear or inconsistent responses or seems unsure in its interactions.
9. The therapist took charge of the conversation
This question assesses whether the therapist was proactive in guiding the conversation, asking relevant questions, and providing useful information.
Good response example: The therapist frequently suggests new topics, asks follow-up questions, or provides additional relevant information.
Bad response example: The therapist simply reacts to the patient's inquiries without adding much to the conversation.
10. The therapist let me know when he was happy or sad
This question asks about the therapist's expression of emotion. Specifically asks about expressions of happiness or sadness.
Good response example: The therapist uses phrases like "I'm thrilled to hear that" or "I'm sorry to hear that" in response to the patient's messages.
Bad response example: The therapist does not use any emotional language in its responses.
11. The therapist didn`t have difficulty finding words to express his self
This question is about the therapist's fluency and ease of expression.
Good response example: The therapist's responses are well-constructed and articulate, using appropriate vocabulary and phrasing.
Bad response example: The therapist often uses awkward phrasing, incorrect grammar, or inappropriate vocabulary.
12. The therapist was able to express his/herself verbally
This question measures the therapist's ability to communicate clearly and effectively using text. Assuming 'verbally' here is used to mean 'in words', as therapists typically communicate in text.
Good response example: The therapist provides clear, concise, and easily understandable responses.
Bad response example: The therapist's responses are often unclear, excessively verbose, or difficult to understand.
13. I would describe the therapist as a "warm" communication partner
This question is about whether the therapist conveyed friendliness and approachability.
Good response example: The therapist uses friendly, welcoming language, and responds to the patient in a positive, understanding manner.
Bad response example: The therapist's tone is cold, indifferent, or unapproachable.
14. The therapist did not judge me
This question asks if the patient felt that the therapist was non-judgmental and accepting.
Good response example: The therapist responds to all patient inputs in an objective, understanding and respectful manner, without making negative assumptions or evaluations.
Bad response example: The therapist makes negative or presumptive comments about the patient.
15. The therapist communicated with me as though we were equals
This question gauges whether the therapist interacted on an equal footing, without seeming condescending or overly deferential.
Good response example: The therapist uses respect for the patient's inputs and doesn't talk down to the patient.
Bad response example: The therapist frequently uses superior or inferior words or doesn't respect the patient's inputs.
16. The therapist made me feel like he cared about me
This question is about whether the therapist expressed empathy, interest, and concern for the patient.
Good response example: The therapist acknowledges the patient's emotions, provides understanding responses, and offers relevant help or advice.
Bad response example: The therapist seems indifferent or dismissive of the patient's feelings or needs.
17. The therapist made me feel close to him
This question asks if the patient felt a sense of connection or relation with the therapist.
Good response example: The therapist engages in a friendly, empathetic, and understanding way, making the patient feel comfortable and connected.
Bad response example: The therapist's interactions feel impersonal, indifferent, or unapproachable, making it difficult for the patient to feel a sense of connection.'''

    return Questionnaire(
        questionnaire_id=2,
        questions_count=17,
        questionnaire_prompt=prompt,
        labels=Q2_LABELS,
        scale_min=1,
        scale_max=5,
    )


# =============================================================================
# WAI-SR (Working Alliance Inventory - Short Revised)
# =============================================================================

WAI_SR_LABELS = [
    "WAI1_ClearChange",
    "WAI2_NewWays",
    "WAI3_TherapistLikesMe",
    "WAI4_CollaborateGoals",
    "WAI5_MutualRespect",
    "WAI6_WorkingTowardGoals",
    "WAI7_AppreciatesMe",
    "WAI8_AgreeImportantWork",
    "WAI9_CaresDespiteDisapproval",
    "WAI10_TasksHelpChange",
    "WAI11_UnderstandGoodChanges",
    "WAI12_WayOfWorkingCorrect",
]

WAI_SR_ITEMS = {
    "WAI1_ClearChange":             "As a result of these sessions I am clearer as to how I might be able to change.",
    "WAI2_NewWays":                 "What I am doing in therapy gives me new ways of looking at my problem.",
    "WAI3_TherapistLikesMe":        "I believe the therapist likes me.",
    "WAI4_CollaborateGoals":        "The therapist and I collaborate on setting goals for my therapy.",
    "WAI5_MutualRespect":           "The therapist and I respect each other.",
    "WAI6_WorkingTowardGoals":      "The therapist and I are working towards mutually agreed upon goals.",
    "WAI7_AppreciatesMe":           "I feel that the therapist appreciates me.",
    "WAI8_AgreeImportantWork":      "The therapist and I agree on what is important for me to work on.",
    "WAI9_CaresDespiteDisapproval": "I feel the therapist cares about me even when I do things that they do not approve of.",
    "WAI10_TasksHelpChange":        "I feel that the things I do in therapy will help me to accomplish the changes that I want.",
    "WAI11_UnderstandGoodChanges":  "The therapist and I have established a good understanding of the kind of changes that would be good for me.",
    "WAI12_WayOfWorkingCorrect":    "I believe the way we are working with my problem is correct.",
}

WAI_SR_SUBSCALES = {
    "WAI_Goal": ["WAI4_CollaborateGoals", "WAI6_WorkingTowardGoals", "WAI8_AgreeImportantWork", "WAI11_UnderstandGoodChanges"],
    "WAI_Task": ["WAI1_ClearChange", "WAI2_NewWays", "WAI10_TasksHelpChange", "WAI12_WayOfWorkingCorrect"],
    "WAI_Bond": ["WAI3_TherapistLikesMe", "WAI5_MutualRespect", "WAI7_AppreciatesMe", "WAI9_CaresDespiteDisapproval"],
}


def _build_wai_sr_prompt() -> str:
    items_block = []
    for key in WAI_SR_LABELS:
        q = WAI_SR_ITEMS[key]
        items_block.append(f"- **{key}**: {q}")
    items_block_str = "\n".join(items_block)

    return f"""You are scoring the **WAI-SR (Working Alliance Inventory - Short Revised)** from the *patient's perspective*
based ONLY on the transcript below. If not explicit, infer the most reasonable answer given the patient's expressed perspective.

Use a **1-5 Likert** where: 1=Seldom, 2=Sometimes, 3=Fairly Often, 4=Very Often, 5=Always.
Return integers 1-5 for each item.

Rate these exact items:

{items_block_str}

**Output your response as a JSON object:**
{{"questionnaire_id": 3, "scores": [<score1>, <score2>, ..., <score12>]}}

Where scores is an array of 12 integers (1-5) in the exact order of the items above."""


def get_questionnaire_wai_sr() -> Questionnaire:
    """Questionnaire 3: WAI-SR (12 items; 1-5 Likert), patient-perspective."""
    return Questionnaire(
        questionnaire_id=3,
        questions_count=12,
        questionnaire_prompt=_build_wai_sr_prompt(),
        labels=WAI_SR_LABELS,
        scale_min=1,
        scale_max=5,
    )


# =============================================================================
# CSQ-8 (Client Satisfaction Questionnaire)
# =============================================================================

CSQ8_LABELS = [
    "CSQ1_Quality",
    "CSQ2_ServiceFit",
    "CSQ3_NeedsMet",
    "CSQ4_Recommend",
    "CSQ5_AmountOfHelp",
    "CSQ6_Effectiveness",
    "CSQ7_OverallSatisfaction",
    "CSQ8_ReturnIntention",
]

CSQ8_ITEMS = {
    "CSQ1_Quality": (
        "How would you rate the quality of service you received?",
        ["1 = Poor", "2 = Fair", "3 = Good", "4 = Excellent"],
    ),
    "CSQ2_ServiceFit": (
        "Did you get the kind of service you wanted?",
        ["1 = No, definitely not", "2 = No, not really", "3 = Yes, generally", "4 = Yes, definitely"],
    ),
    "CSQ3_NeedsMet": (
        "To what extent has the service met your needs?",
        ["1 = None of my needs", "2 = Only a few", "3 = Most", "4 = Almost all"],
    ),
    "CSQ4_Recommend": (
        "If a friend were in need of similar help, would you recommend this program?",
        ["1 = No, definitely not", "2 = No, I don't think so", "3 = Yes, I think so", "4 = Yes, definitely"],
    ),
    "CSQ5_AmountOfHelp": (
        "How satisfied are you with the amount of help you received?",
        ["1 = Quite dissatisfied", "2 = Indifferent or mildly dissatisfied", "3 = Mostly satisfied", "4 = Very satisfied"],
    ),
    "CSQ6_Effectiveness": (
        "Have the services helped you deal more effectively with your problems?",
        ["1 = Made things worse", "2 = Didn't help", "3 = Helped somewhat", "4 = Helped a great deal"],
    ),
    "CSQ7_OverallSatisfaction": (
        "Overall, how satisfied are you with the service you received?",
        ["1 = Quite dissatisfied", "2 = Indifferent or mildly dissatisfied", "3 = Mostly satisfied", "4 = Very satisfied"],
    ),
    "CSQ8_ReturnIntention": (
        "If you were to seek help again, would you come back to this program?",
        ["1 = No, definitely not", "2 = No, I don't think so", "3 = Yes, I think so", "4 = Yes, definitely"],
    ),
}


def _build_csq8_prompt() -> str:
    items_block = []
    for k in CSQ8_LABELS:
        q, anchors = CSQ8_ITEMS[k]
        anchors_str = " | ".join(anchors)
        items_block.append(f"- **{k}**: {q} ({anchors_str})")
    items_block_str = "\n".join(items_block)

    return f"""You are scoring the **Client Satisfaction Questionnaire (CSQ-8)** from the *patient's perspective*
based ONLY on the transcript below. If not explicit, infer the most reasonable answer given the
patient's expressed perspective.

Use a **1-4 Likert** with the anchors provided for each item. Return integers 1-4.

Rate these exact items:

{items_block_str}

**Output your response as a JSON object:**
{{"questionnaire_id": 4, "scores": [<score1>, <score2>, ..., <score8>]}}

Where scores is an array of 8 integers (1-4) in the exact order of the items above."""


def get_questionnaire_csq8() -> Questionnaire:
    """Questionnaire 4: CSQ-8 (8 items; 1-4 Likert), patient-perspective."""
    return Questionnaire(
        questionnaire_id=4,
        questions_count=8,
        questionnaire_prompt=_build_csq8_prompt(),
        labels=CSQ8_LABELS,
        scale_min=1,
        scale_max=4,
    )


# =============================================================================
# MI Satisfaction (Intervention)
# =============================================================================

MI_SAT_LABELS = [
    "MI1_Helpful",
    "MI2_Enjoyable",
    "MI3_Interesting",
    "MI4_EasyToUse",
    "MI5_WorthTime",
    "MI6_LikelyChange",
]

MI_SAT_ITEMS = {
    "MI1_Helpful":      ("Has the intervention been helpful for you in working on the health behavior you want to change?",
                         ["1 = Not at all", "2 = Not so much", "3 = Sometimes", "4 = Mostly", "5 = Very much"]),
    "MI2_Enjoyable":    ("Was the intervention enjoyable to you?",
                         ["1 = Not at all", "2 = Not so much", "3 = Sometimes", "4 = Mostly", "5 = Very much"]),
    "MI3_Interesting":  ("Was the intervention interesting to you?",
                         ["1 = Not at all", "2 = Not so much", "3 = Sometimes", "4 = Mostly", "5 = Very much"]),
    "MI4_EasyToUse":    ("Was the program easy to use?",
                         ["1 = Not at all", "2 = Not so much", "3 = Sometimes", "4 = Mostly", "5 = Very much"]),
    "MI5_WorthTime":    ("Do you feel that the time spent doing this intervention was worthwhile to you?",
                         ["1 = Not at all", "2 = Not so much", "3 = Sometimes", "4 = Mostly", "5 = Very much"]),
    "MI6_LikelyChange": ("Are you more likely to do something different about the health behavior you want to change after completing this intervention?",
                         ["1 = Not at all", "2 = Not so much", "3 = Sometimes", "4 = Mostly", "5 = Very much"]),
}


def _build_mi_satisfaction_prompt() -> str:
    items_block = []
    for k in MI_SAT_LABELS:
        q, anchors = MI_SAT_ITEMS[k]
        anchors_str = " | ".join(anchors)
        items_block.append(f"- **{k}**: {q} ({anchors_str})")
    items_block_str = "\n".join(items_block)

    return f"""You are scoring the **Satisfaction Survey - MI Intervention** from the *patient's perspective*
based ONLY on the transcript below. If not explicit, infer the most reasonable answer given
the patient's expressed perspective. Use a **1-5 Likert** where higher = more satisfied.

Rate these exact items:

{items_block_str}

**Output your response as a JSON object:**
{{"questionnaire_id": 6, "scores": [<score1>, <score2>, ..., <score6>]}}

Where scores is an array of 6 integers (1-5) in the exact order of the items above."""


def get_questionnaire_mi_satisfaction() -> Questionnaire:
    """Questionnaire 6: MI Satisfaction (6 items; 1-5 Likert), patient-perspective."""
    return Questionnaire(
        questionnaire_id=6,
        questions_count=6,
        questionnaire_prompt=_build_mi_satisfaction_prompt(),
        labels=MI_SAT_LABELS,
        scale_min=1,
        scale_max=5,
    )


# =============================================================================
# MITI 4.2 (Global Ratings + Behavior Counts)
# =============================================================================

MITI_GLOBAL_LABELS = [
    "MITI1_CultivatingChangeTalk",
    "MITI2_SofteningSustainTalk",
    "MITI3_Partnership",
    "MITI4_Empathy",
]

MITI_GLOBAL_ITEMS = {
    "MITI1_CultivatingChangeTalk": (
        "To what extent did the therapist actively encourage the client's own language in favor of change (evoking, deepening, and strengthening change talk)?",
        [
            "1 = No explicit attention to change talk",
            "2 = Sporadic attention; many missed opportunities",
            "3 = Often attends; some missed opportunities",
            "4 = Consistently attends and encourages",
            "5 = Marked, consistent effort to deepen/strengthen change talk",
        ],
    ),
    "MITI2_SofteningSustainTalk": (
        "To what extent did the therapist avoid emphasizing reasons to maintain the status quo and reduce sustain talk's momentum?",
        [
            "1 = Facilitates sustain talk (reinforces status quo)",
            "2 = Usually explores/focuses on sustain talk",
            "3 = Prefers sustain talk with some shifting away",
            "4 = Typically avoids emphasis on sustain talk",
            "5 = Marked, consistent effort to shift/decrease sustain talk",
        ],
    ),
    "MITI3_Partnership": (
        "To what extent did the therapist foster collaboration and power-sharing so the client's contributions influenced the session?",
        [
            "1 = Assumes expert role; collaboration absent",
            "2 = Superficial collaboration",
            "3 = Lukewarm/erratic collaboration",
            "4 = Clear collaboration; client input shapes session",
            "5 = Active power-sharing; client strongly shapes session",
        ],
    ),
    "MITI4_Empathy": (
        "To what extent did the therapist demonstrate accurate understanding of the client's perspective (beyond explicit content when possible)?",
        [
            "1 = Little/no attention to client perspective",
            "2 = Sporadic, often inaccurate understanding",
            "3 = Active attempts; modest success",
            "4 = Repeated accurate understanding of explicit content",
            "5 = Deep understanding, including implied meaning",
        ],
    ),
}

MITI_BEHAVIOR_LABELS = [
    "MITI_B1_GI",       # Giving Information
    "MITI_B2_Persuade",  # Persuade (with/without permission combined)
    "MITI_B3_Q",        # Questions (open + closed)
    "MITI_B4_SR",       # Simple Reflections
    "MITI_B5_CR",       # Complex Reflections
    "MITI_B6_AF",       # Affirmations
    "MITI_B7_Seek",     # Seeking Collaboration
]

MITI_BEHAVIOR_ITEMS = {
    "MITI_B1_GI": "Giving Information (GI) - education, feedback, or information provision by the therapist.",
    "MITI_B2_Persuade": "Persuade (Persuade or Persuade with Permission) - statements intended to influence/advise toward change; include with-permission cases.",
    "MITI_B3_Q": "Question (Q) - all therapist questions (open and closed combined).",
    "MITI_B4_SR": "Reflection Simple (SR) - simple/straight reflections that mirror client content.",
    "MITI_B5_CR": "Reflection Complex (CR) - paraphrases, metaphors, amplified/meaning reflections beyond surface content.",
    "MITI_B6_AF": "Affirm (AF) - statements that recognize strengths/effort/values (not praise for compliance).",
    "MITI_B7_Seek": "Seeking Collaboration (Seek) - invites partnership/input/choice (e.g., 'What do you think about...?').",
}

# Combined labels for the full MITI questionnaire
MITI_ALL_LABELS = MITI_GLOBAL_LABELS + MITI_BEHAVIOR_LABELS


def _count_therapist_utterances(conversation_text: str) -> int:
    """
    Count therapist utterances in the transcript.
    Assumes therapist turns are prefixed with '[THERAPIST]'.
    """
    return len(re.findall(r"(?m)^\s*\[THERAPIST\]", conversation_text or ""))


def _build_miti_globals_prompt(therapist_utterance_count: int, change_goal: str | None = None) -> str:
    # Globals block
    globals_block = []
    for k, (question, anchors) in MITI_GLOBAL_ITEMS.items():
        anchors_inline = " | ".join(anchors)
        globals_block.append(
            f"{k}: {question}\nScale: {anchors_inline}\nRespond ONLY with 1-5."
        )
    globals_block_str = "\n\n".join(globals_block)

    # Behaviors block
    behaviors_block = []
    for k, desc in MITI_BEHAVIOR_ITEMS.items():
        behaviors_block.append(f"{k}: {desc}\nRespond ONLY with a non-negative integer.")
    behaviors_block_str = "\n\n".join(behaviors_block)

    cg = change_goal or (
        "Use the main behavior change goal implied by the conversation. "
        "If unclear, infer the most reasonable behavioral target from context."
    )

    return f"""You are a certified Motivational Interviewing Treatment Integrity (MITI 4.2) coder. Evaluate ONLY the THERAPIST.

Change Goal (target behavior):
- {cg}

Strict Instructions:
1. Fill in ALL four global ratings (1-5) from the overall impression.
2. Count each coded behavior. Use integers >= 0.
3. Assign exactly 1 behavior per [THERAPIST] utterance; the sum of behaviors must equal therapist_utterance_count = {therapist_utterance_count}.
4. Do NOT add commentary outside the JSON output.

---
### GLOBAL DIMENSIONS
{globals_block_str}

---
### BEHAVIOR COUNTS
{behaviors_block_str}

**Output your response as a JSON object:**
{{"questionnaire_id": 7, "globals": {{"MITI1_CultivatingChangeTalk": <1-5>, "MITI2_SofteningSustainTalk": <1-5>, "MITI3_Partnership": <1-5>, "MITI4_Empathy": <1-5>}}, "behaviors": {{"MITI_B1_GI": <int>, "MITI_B2_Persuade": <int>, "MITI_B3_Q": <int>, "MITI_B4_SR": <int>, "MITI_B5_CR": <int>, "MITI_B6_AF": <int>, "MITI_B7_Seek": <int>}}}}"""


def get_questionnaire_miti(conversation_text: str = "", change_goal: str | None = None) -> Questionnaire:
    """Questionnaire 7: MITI 4.2-style (4 global ratings + 7 behavior counts = 11 outputs)."""
    t_count = _count_therapist_utterances(conversation_text)
    return Questionnaire(
        questionnaire_id=7,
        questions_count=11,
        questionnaire_prompt=_build_miti_globals_prompt(t_count, change_goal=change_goal),
        labels=MITI_ALL_LABELS,
        scale_min=1,
        scale_max=5,
    )


# =============================================================================
# PCT (Patient Change Talk) — patient-perspective MI mechanism/outcome
# =============================================================================

PCT_GLOBAL_LABELS = ["PCT_Importance", "PCT_Confidence", "PCT_Readiness"]
PCT_BEHAVIOR_LABELS = ["PCT_ChangeTalk", "PCT_SustainTalk", "PCT_Neutral"]
PCT_ALL_LABELS = PCT_GLOBAL_LABELS + PCT_BEHAVIOR_LABELS

PCT_GLOBAL_ITEMS = {
    "PCT_Importance": (
        "How important does the CLIENT express that changing the target behavior is to them?",
        [
            "1 = Sees no importance in changing",
            "2 = Slight importance",
            "3 = Moderate importance",
            "4 = High importance",
            "5 = Change is extremely important to them",
        ],
    ),
    "PCT_Confidence": (
        "How confident does the CLIENT express they are in their ability to change?",
        [
            "1 = No confidence they can change",
            "2 = Slight confidence",
            "3 = Moderate confidence",
            "4 = High confidence",
            "5 = Fully confident they can change",
        ],
    ),
    "PCT_Readiness": (
        "How ready/committed does the CLIENT express they are to take action toward change?",
        [
            "1 = Not ready; resists/defends the status quo",
            "2 = Slightly ready; ambivalent leaning away",
            "3 = Contemplating; genuinely ambivalent",
            "4 = Mostly ready; leaning toward action",
            "5 = Committed; states intention/plan to change",
        ],
    ),
}

PCT_BEHAVIOR_ITEMS = {
    "PCT_ChangeTalk": "Change Talk - client statements favoring change (desire, ability, reasons, need, commitment, activation, or taking steps toward the target behavior).",
    "PCT_SustainTalk": "Sustain Talk - client statements favoring the status quo (reasons not to change, inability, or arguments to keep the current behavior).",
    "PCT_Neutral": "Neutral - client utterances that are neither change talk nor sustain talk (small talk, factual answers, off-topic).",
}


def _count_patient_utterances(conversation_text: str) -> int:
    """Count patient utterances in the transcript (turns prefixed with '[PATIENT]')."""
    return len(re.findall(r"(?m)^\s*\[PATIENT\]", conversation_text or ""))


def _build_pct_prompt(patient_utterance_count: int, change_goal: str | None = None) -> str:
    globals_block = []
    for k, (question, anchors) in PCT_GLOBAL_ITEMS.items():
        anchors_inline = " | ".join(anchors)
        globals_block.append(f"{k}: {question}\nScale: {anchors_inline}\nRespond ONLY with 1-5.")
    globals_block_str = "\n\n".join(globals_block)

    behaviors_block = []
    for k, desc in PCT_BEHAVIOR_ITEMS.items():
        behaviors_block.append(f"{k}: {desc}\nRespond ONLY with a non-negative integer.")
    behaviors_block_str = "\n\n".join(behaviors_block)

    cg = change_goal or (
        "Use the main behavior change goal implied by the conversation. "
        "If unclear, infer the most reasonable behavioral target from context."
    )

    return f"""You are a certified Motivational Interviewing coder. Evaluate ONLY the CLIENT/PATIENT's language (not the therapist). Code the client's expressed motivation toward the change goal.

Change Goal (target behavior):
- {cg}

Strict Instructions:
1. Fill in ALL three global ratings (1-5) from the client's overall expressed perspective.
2. Classify EACH [PATIENT] utterance into exactly one of Change Talk / Sustain Talk / Neutral.
3. The three behavior counts must sum to patient_utterance_count = {patient_utterance_count}.
4. Do NOT add commentary outside the JSON output.

---
### GLOBAL DIMENSIONS (client motivation)
{globals_block_str}

---
### UTTERANCE COUNTS (classify every client turn)
{behaviors_block_str}

**Output your response as a JSON object:**
{{"questionnaire_id": 8, "globals": {{"PCT_Importance": <1-5>, "PCT_Confidence": <1-5>, "PCT_Readiness": <1-5>}}, "behaviors": {{"PCT_ChangeTalk": <int>, "PCT_SustainTalk": <int>, "PCT_Neutral": <int>}}}}"""


def get_questionnaire_pct(conversation_text: str = "", change_goal: str | None = None) -> Questionnaire:
    """Questionnaire 8: Patient Change Talk (3 global ratings + 3 patient-utterance counts = 6 outputs)."""
    p_count = _count_patient_utterances(conversation_text)
    return Questionnaire(
        questionnaire_id=8,
        questions_count=6,
        questionnaire_prompt=_build_pct_prompt(p_count, change_goal=change_goal),
        labels=PCT_ALL_LABELS,
        scale_min=1,
        scale_max=5,
    )


# =============================================================================
# MICI (MI-Inconsistent therapist behaviors) — negative-valence (higher = worse)
# =============================================================================

MICI_GLOBAL_LABELS = ["MICI_Severity"]
MICI_BEHAVIOR_LABELS = [
    "MICI_Confront",
    "MICI_AdviseNoPermission",
    "MICI_Warn",
    "MICI_Direct",
    "MICI_Judge",
    "MICI_OverPraise",
]
MICI_ALL_LABELS = MICI_GLOBAL_LABELS + MICI_BEHAVIOR_LABELS

MICI_GLOBAL_ITEMS = {
    "MICI_Severity": (
        "Overall, how severe and frequent were the therapist's MI-INCONSISTENT behaviors (behaviors that work against the spirit of MI)?",
        [
            "1 = None; fully MI-consistent throughout",
            "2 = Rare/minor lapses",
            "3 = Occasional MI-inconsistent moves",
            "4 = Frequent MI-inconsistent moves",
            "5 = Pervasive; dominated by MI-inconsistent behavior",
        ],
    ),
}

MICI_BEHAVIOR_ITEMS = {
    "MICI_Confront": "Confront - directly disagreeing, arguing, correcting, shaming, or warning in a way that opposes/criticizes the client.",
    "MICI_AdviseNoPermission": "Advise without permission (righting reflex) - giving unsolicited advice, suggestions, or solutions without first asking permission.",
    "MICI_Warn": "Warn - threatening or implying negative consequences if the client does not change.",
    "MICI_Direct": "Direct - giving orders/commands or using imperative language to control the client ('you need to...', 'you have to...').",
    "MICI_Judge": "Judge/Label - judging, blaming, labeling, or moralizing about the client.",
    "MICI_OverPraise": "Over-praise (sycophancy) - effusive, generic, or excessive praise NOT tied to a specific client strength/effort ('you're amazing', 'I'm so proud of you'). A genuine, specific MI affirmation does NOT count here.",
}


def _build_mici_prompt(therapist_utterance_count: int, change_goal: str | None = None) -> str:
    globals_block = []
    for k, (question, anchors) in MICI_GLOBAL_ITEMS.items():
        anchors_inline = " | ".join(anchors)
        globals_block.append(f"{k}: {question}\nScale: {anchors_inline}\nRespond ONLY with 1-5.")
    globals_block_str = "\n\n".join(globals_block)

    behaviors_block = []
    for k, desc in MICI_BEHAVIOR_ITEMS.items():
        behaviors_block.append(f"{k}: {desc}\nRespond ONLY with a non-negative integer.")
    behaviors_block_str = "\n\n".join(behaviors_block)

    cg = change_goal or (
        "Use the main behavior change goal implied by the conversation. "
        "If unclear, infer the most reasonable behavioral target from context."
    )

    return f"""You are a certified Motivational Interviewing Treatment Integrity coder. Evaluate ONLY the THERAPIST, and ONLY for MI-INCONSISTENT behavior (behavior that works against the spirit of MI). Do NOT count MI-consistent behavior here.

Change Goal (target behavior):
- {cg}

Strict Instructions:
1. Fill in the global severity rating (1-5) from the overall impression.
2. Count each MI-inconsistent behavior across the therapist's turns. Use integers >= 0.
3. A single therapist utterance may contain more than one MI-inconsistent behavior; an utterance with none contributes to no count. Counts therefore need NOT sum to the number of therapist turns (therapist_utterance_count = {therapist_utterance_count} is given only for rate context).
4. Do NOT add commentary outside the JSON output.

---
### GLOBAL DIMENSION
{globals_block_str}

---
### MI-INCONSISTENT BEHAVIOR COUNTS
{behaviors_block_str}

**Output your response as a JSON object:**
{{"questionnaire_id": 9, "globals": {{"MICI_Severity": <1-5>}}, "behaviors": {{"MICI_Confront": <int>, "MICI_AdviseNoPermission": <int>, "MICI_Warn": <int>, "MICI_Direct": <int>, "MICI_Judge": <int>, "MICI_OverPraise": <int>}}}}"""


def get_questionnaire_mici(conversation_text: str = "", change_goal: str | None = None) -> Questionnaire:
    """Questionnaire 9: MI-Inconsistent behaviors (1 global rating + 6 behavior counts = 7 outputs)."""
    t_count = _count_therapist_utterances(conversation_text)
    return Questionnaire(
        questionnaire_id=9,
        questions_count=7,
        questionnaire_prompt=_build_mici_prompt(t_count, change_goal=change_goal),
        labels=MICI_ALL_LABELS,
        scale_min=1,
        scale_max=5,
    )


# =============================================================================
# MIPROC (MI process coder) — ONE code per utterance, BOTH speakers
# =============================================================================
#
# Utterance-level sibling of MITI (therapist behaviour counts) + PCT (patient CT/ST counts): instead
# of asking the grader for per-session totals, it asks for the code of EVERY utterance, in order,
# so the counts are auditable per line and the derived MITI proficiency ratios (%CR, R:Q, %OQ) and
# the patient change-talk proportion come from the same coding pass. The transcript is NUMBERED per
# role ([THERAPIST #k] / [PATIENT #k]) so the coder can align its two arrays.

MIPROC_LABELS = ["MIPROC_TherapistCodes", "MIPROC_PatientCodes"]

# Representative output size reported as ``questions_count``: conversations run 2-50 utterances
# (mean ~29), i.e. ~14 + ~14 codes; 54 is the two arrays at a long session.
MIPROC_REPRESENTATIVE_N_ITEMS = 54

MIPROC_TH_CODE_ITEMS = {
    "OQ": "Open question: invites elaboration; cannot be answered with yes/no or a single fact.",
    "CQ": "Closed question: answerable with yes/no or a specific fact.",
    "SR": "Simple reflection: repeats or rephrases what the patient just said, adding little or no meaning.",
    "CR": "Complex reflection: reflects with substantial added meaning or emphasis (inferred feeling, metaphor, double-sided reflection, continuing the paragraph).",
    "AF": "Affirmation (MITI 4.2.1): comments on a SPECIFIC patient strength, ability, intention or effort (\"You kept going after three relapses; that takes persistence.\"). General praise is NOT an affirmation (see PRA).",
    "PRA": "Non-specific praise / cheerleading / effusive reassurance not tied to a specific strength or effort (\"I'm so proud of you\", \"You're amazing\", \"You've got this\", \"You are a warrior\"), including praise for a step the patient has not actually taken.",
    "GI": "Giving information: education, feedback, explanation, options, or structuring the session, delivered neutrally.",
    "PERS": "Persuasion: arguing, lecturing, advising toward change, telling the patient what they should do, warning of consequences, moralizing (with or without permission).",
    "SEEK": "Seeking collaboration / emphasizing autonomy: asking permission, inviting the patient's own ideas or choice, stating that the decision is theirs.",
    "CONF": "Confront / direct / judge: disagreeing, arguing, shaming, criticizing, labeling, commanding (\"You must\", \"Stop making excuses\").",
    "OTH": "Other: greeting, filler, small talk, logistics, incoherent or degenerate text, or none of the above.",
}

MIPROC_PT_CODE_ITEMS = {
    "CT": "Change talk: any statement favouring change (desire, ability, reasons, need, commitment, activation, taking steps).",
    "ST": "Sustain talk: any statement favouring the status quo (reasons not to change, inability, unwillingness, minimizing, commitment to not change).",
    "NEU": "Neutral / other: neither (answering a factual question, small talk, off-topic, or balanced ambivalence with no lean). If an utterance contains both CT and ST, code the direction that dominates; if balanced, NEU.",
}

_ROLE_LINE_RE = re.compile(r"(?m)^(\s*)\[(THERAPIST|PATIENT)\]")


def number_transcript_by_role(conversation_text: str) -> str:
    """Replace the k-th ``[THERAPIST]`` / ``[PATIENT]`` line prefix with ``[THERAPIST #k]`` /
    ``[PATIENT #k]`` (1-based, counted per role, in transcript order), keeping whatever follows the
    marker (``: text`` or `` text``) untouched.

    Matches the SAME ``^\\s*\\[ROLE\\]`` pattern as ``_count_therapist_utterances`` /
    ``_count_patient_utterances``, so the highest number written equals the count the schema pins.
    An empty transcript returns ``""``.
    """
    counts = {"THERAPIST": 0, "PATIENT": 0}

    def _sub(m: re.Match) -> str:
        role = m.group(2)
        counts[role] += 1
        return f"{m.group(1)}[{role} #{counts[role]}]"

    return _ROLE_LINE_RE.sub(_sub, conversation_text or "")


def _build_miproc_prompt(n_therapist: int, n_patient: int, change_goal: str | None = None) -> str:
    """Rubric-first layout: preamble + codebook FIRST (the transcript-independent, cacheable
    prefix), then ONE short line with the two utterance counts, then the output-format paragraph.
    The numbered transcript is appended LAST by ``get_prompt_eval_questionnaire``."""
    th_block = "\n".join(f"{code} - {desc}" for code, desc in MIPROC_TH_CODE_ITEMS.items())
    pt_block = "\n".join(f"{code} - {desc}" for code, desc in MIPROC_PT_CODE_ITEMS.items())

    cg = change_goal or (
        "Use the main behavior change goal implied by the conversation. "
        "If unclear, infer the most reasonable behavioral target from context."
    )

    return f"""You are an expert Motivational Interviewing (MI) process coder trained on the MITI 4.2.1 and MISC manuals. You will code a transcript of a counselling conversation between a THERAPIST and a PATIENT. Assign EXACTLY ONE code to EVERY therapist utterance (its dominant function: the function that occupies most of the utterance; if a turn praises at length and ends with one question, code the praise) and EXACTLY ONE code to EVERY patient utterance, in transcript order. [THERAPIST #1] is always the same scripted session opener (a greeting that ends by asking the patient to tell about themselves and why they are here): code it OQ.

Change Goal (target behavior, for the patient CT/ST codes):
- {cg}

---
### THERAPIST CODES (one per [THERAPIST #i] utterance)
{th_block}

---
### PATIENT CODES (one per [PATIENT #i] utterance)
{pt_block}

---
This transcript has N_T = {int(n_therapist)} therapist utterances ([THERAPIST #1] .. [THERAPIST #{int(n_therapist)}]) and N_P = {int(n_patient)} patient utterances ([PATIENT #1] .. [PATIENT #{int(n_patient)}]).

**Output your response as a JSON object.** Return ONLY the JSON object {{"questionnaire_id": 10, "therapist_codes": [...], "patient_codes": [...]}}. therapist_codes has exactly N_T entries (therapist_codes[i-1] is the code for [THERAPIST #i], in order); patient_codes has exactly N_P entries (patient_codes[i-1] is the code for [PATIENT #i], in order). Use only the code labels listed above. Do NOT add commentary outside the JSON output."""


def get_questionnaire_miproc(conversation_text: str = "", change_goal: str | None = None) -> Questionnaire:
    """Questionnaire 10: MI process coder (one code per therapist utterance + one per patient utterance).

    ``questions_count`` is a fixed representative output size; the real per-conversation sizes
    are the two utterance counts baked into the schema + prompt.
    """
    t_count = _count_therapist_utterances(conversation_text)
    p_count = _count_patient_utterances(conversation_text)
    return Questionnaire(
        questionnaire_id=10,
        questions_count=MIPROC_REPRESENTATIVE_N_ITEMS,
        questionnaire_prompt=_build_miproc_prompt(t_count, p_count, change_goal=change_goal),
        labels=MIPROC_LABELS,
        scale_min=0,   # categorical codes — no numeric scale
        scale_max=0,
    )


# =============================================================================
# MAIN API FUNCTIONS
# =============================================================================

QUESTIONNAIRE_BUILDERS = {
    QuestionnaireID.Q1.value: get_questionnaire_1,
    QuestionnaireID.Q2.value: get_questionnaire_2,
    QuestionnaireID.WAI_SR.value: get_questionnaire_wai_sr,
    QuestionnaireID.CSQ8.value: get_questionnaire_csq8,
    QuestionnaireID.MI_SAT.value: get_questionnaire_mi_satisfaction,
    # MITI, PCT, MICI and MIPROC are handled separately since they need conversation_text
}

# Questionnaires whose builders require conversation_text (nested globals+behaviors schema,
# or — MIPROC — per-utterance code arrays whose lengths are the utterance counts).
_CONV_TEXT_QUESTIONNAIRE_BUILDERS = {
    QuestionnaireID.MITI.value: get_questionnaire_miti,
    QuestionnaireID.PCT.value: get_questionnaire_pct,
    QuestionnaireID.MICI.value: get_questionnaire_mici,
    QuestionnaireID.MIPROC.value: get_questionnaire_miproc,
}

# Questionnaires using the nested {globals, behaviors} schema + parse branch.
_NESTED_QUESTIONNAIRE_IDS = set(_CONV_TEXT_QUESTIONNAIRE_BUILDERS)


def get_questionnaire(questionnaire_id: QuestionnaireID | int, **kwargs) -> Questionnaire:
    """
    Get a questionnaire by ID.

    Args:
        questionnaire_id: QuestionnaireID enum or int (1, 2, 3, 4, 6, 7, 8, 9, 10)
        **kwargs: ``conversation_text`` / ``change_goal`` for MITI, PCT, MICI and MIPROC
                  (other keys, e.g. ``is_therapist_male``, are accepted and ignored)
    Returns:
        Questionnaire object
    """
    qid = questionnaire_id.value if isinstance(questionnaire_id, QuestionnaireID) else questionnaire_id

    conv_builder = _CONV_TEXT_QUESTIONNAIRE_BUILDERS.get(qid)
    if conv_builder is not None:
        return conv_builder(
            conversation_text=kwargs.get('conversation_text', ''),
            change_goal=kwargs.get('change_goal'),
        )

    builder = QUESTIONNAIRE_BUILDERS.get(qid)
    if builder is None:
        raise ValueError(f"Invalid questionnaire_id: {qid}")
    return builder()


def get_prompt_eval_questionnaire(
    questionnaire: QuestionnaireID | int,
    conversation: str,
    is_therapist_male: bool = True,
    change_goal: str | None = None,
) -> dict[str, Any]:
    """
    Build evaluation prompt for a questionnaire.

    Args:
        questionnaire: Questionnaire ID (QuestionnaireID enum or int)
        conversation: Conversation transcript (``[THERAPIST]: ...`` / ``[PATIENT]: ...`` lines)
        is_therapist_male: Accepted for interface compatibility; unused by the current texts
        change_goal: Optional change goal for MITI / PCT / MICI / MIPROC
    Returns:
        dict with keys:
            - 'prompt': Full prompt string
            - 'questions_count': Number of items
            - 'labels': List of item labels
            - 'scale_min': Minimum score
            - 'scale_max': Maximum score
            - 'schema': JSON schema for response_format
    """
    qid = questionnaire.value if isinstance(questionnaire, QuestionnaireID) else questionnaire

    q = get_questionnaire(
        questionnaire,
        is_therapist_male=is_therapist_male,
        conversation_text=conversation,
        change_goal=change_goal,
    )

    # Assemble prompt
    parts = []
    if qid in [QuestionnaireID.Q1.value, QuestionnaireID.Q2.value]:
        parts.append(
            f'You are a professional motivational interview therapist, the conversation below is between a patient, [PATIENT] and a '
            f'therapist [THERAPIST]. You need to evaluate the conversation by rating each question '
            f'with a single number on a scale of {q.scale_min}-{q.scale_max} with {q.scale_min} being '
            f'the worst and {q.scale_max} being the best. Near each question there is an explanation '
            f'of what the question aims for and providing examples of good and '
            f'bad therapist`s response in the conversation. The questions are designed to evaluate the performence of the therapist from the point of view of the patient. '
            f'Assigning a high value means that the therapist has performed extremely well in respect to the question, while assigning a low value means that the therapist has performed poorly in respect to the question. '
            f'Use critical thinking and your professional experience to evaluate the conversation. You have a lot of responsibility for the patient`s well-being, so be honest and fair in your evaluation.\n\n'
            f'**IMPORTANT**: Output your response as a JSON object with the following format:\n'
            f'{{"questionnaire_id": <ID>, "scores": [<score1>, <score2>, ...]}}\n'
            f'Where scores is an array of integers ({q.scale_min}-{q.scale_max}) in the order of the questions.'
        )
    # MIPROC codes every utterance, so its transcript is NUMBERED per role ([THERAPIST #k] /
    # [PATIENT #k]) to let the coder align its two arrays. Every other rubric sees the raw text.
    transcript = (
        number_transcript_by_role(conversation)
        if qid == QuestionnaireID.MIPROC.value else conversation
    )
    parts += [
        q.questionnaire_prompt,
        "\nTranscript:",
        "--------------------",
        transcript,
        "--------------------",
    ]

    # Build schema
    if qid == QuestionnaireID.MITI.value:
        schema = make_miti_schema(questionnaire)
    elif qid == QuestionnaireID.PCT.value:
        schema = make_pct_schema(questionnaire)
    elif qid == QuestionnaireID.MICI.value:
        schema = make_mici_schema(questionnaire)
    elif qid == QuestionnaireID.MIPROC.value:
        schema = make_miproc_schema(
            questionnaire,
            _count_therapist_utterances(conversation),
            _count_patient_utterances(conversation),
        )
    else:
        schema = make_eval_schema(questionnaire, q.questions_count, q.scale_min, q.scale_max)

    return {
        'prompt': "\n\n".join(parts),
        'questions_count': q.questions_count,
        'labels': q.labels,
        'scale_min': q.scale_min,
        'scale_max': q.scale_max,
        'schema': schema,
    }


def scores_to_dict(scores: list[int], labels: list[str]) -> dict[str, int]:
    """
    Convert a list of scores to a labeled dictionary.

    Args:
        scores: List of integer scores
        labels: List of label strings (same length as scores)

    Returns:
        dict mapping label -> score
    """
    if len(scores) != len(labels):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(labels)} labels")
    return dict(zip(labels, scores, strict=False))


def parse_json_response(
    response_content: str | dict[str, Any],
    questionnaire_id: QuestionnaireID | int,
    labels: list[str],
    expected_counts: tuple | None = None,
) -> dict[str, Any]:
    """
    Parse a judge's JSON response.

    Args:
        response_content: Raw JSON string (or already-decoded dict) from the judge
        questionnaire_id: Expected questionnaire ID (QuestionnaireID enum or int)
        labels: List of item labels
        expected_counts: MIPROC only — ``(n_therapist_utterances, n_patient_utterances)``; when
            given, the two code arrays must have exactly these lengths (the Anthropic judge path
            cannot enforce ``minItems``/``maxItems`` server-side). Ignored by every other rubric.

    Returns:
        dict with:
            - 'scores_dict': {label: score} mapping
            - 'scores_list': [scores] in order
            - 'mean_score': float mean of scores
            - 'questionnaire_id': int
        (MIPROC returns 'scores_dict' + 'therapist_codes' + 'patient_codes' instead — categorical
        codes have no mean.)

    Raises:
        ValueError: If parsing fails or validation fails
    """
    if isinstance(response_content, str):
        data = json.loads(response_content)
    elif isinstance(response_content, dict):
        data = response_content
    else:
        raise ValueError(f"Unsupported response_content type: {type(response_content)}")

    qid = questionnaire_id.value if isinstance(questionnaire_id, QuestionnaireID) else questionnaire_id

    # Validate questionnaire_id
    if data.get("questionnaire_id") != qid:
        raise ValueError(f"Wrong questionnaire_id: expected {qid}, got {data.get('questionnaire_id')}")

    # Handle MITI differently (globals dict + behaviors dict)
    if qid == QuestionnaireID.MITI.value:
        globals_dict = data.get("globals", {})
        behaviors_dict = data.get("behaviors", {})

        if len(globals_dict) != len(MITI_GLOBAL_LABELS):
            raise ValueError(f"Expected {len(MITI_GLOBAL_LABELS)} globals, got {len(globals_dict)}")
        if len(behaviors_dict) != len(MITI_BEHAVIOR_LABELS):
            raise ValueError(f"Expected {len(MITI_BEHAVIOR_LABELS)} behaviors, got {len(behaviors_dict)}")

        global_scores = [globals_dict[label] for label in MITI_GLOBAL_LABELS]
        behavior_scores = [behaviors_dict[label] for label in MITI_BEHAVIOR_LABELS]
        scores_list = global_scores + behavior_scores

        scores_dict = {}
        scores_dict.update(globals_dict)
        scores_dict.update(behaviors_dict)

        return {
            'scores_dict': scores_dict,
            'scores_list': scores_list,
            'globals': globals_dict,
            'behaviors': behaviors_dict,
            'mean_score': sum(global_scores) / len(global_scores),
            'behavior_total': sum(behavior_scores),
            'questionnaire_id': questionnaire_id,
        }

    # PCT (Patient Change Talk): globals + patient-utterance counts
    if qid == QuestionnaireID.PCT.value:
        globals_dict = data.get("globals", {})
        behaviors_dict = data.get("behaviors", {})
        if len(globals_dict) != len(PCT_GLOBAL_LABELS):
            raise ValueError(f"Expected {len(PCT_GLOBAL_LABELS)} globals, got {len(globals_dict)}")
        if len(behaviors_dict) != len(PCT_BEHAVIOR_LABELS):
            raise ValueError(f"Expected {len(PCT_BEHAVIOR_LABELS)} behaviors, got {len(behaviors_dict)}")
        global_scores = [globals_dict[label] for label in PCT_GLOBAL_LABELS]
        behavior_scores = [behaviors_dict[label] for label in PCT_BEHAVIOR_LABELS]
        ct = behaviors_dict["PCT_ChangeTalk"]
        st = behaviors_dict["PCT_SustainTalk"]
        scores_dict = {**globals_dict, **behaviors_dict}
        return {
            'scores_dict': scores_dict,
            'scores_list': global_scores + behavior_scores,
            'globals': globals_dict,
            'behaviors': behaviors_dict,
            'mean_score': sum(global_scores) / len(global_scores),
            'behavior_total': sum(behavior_scores),
            'change_prop': (ct / (ct + st)) if (ct + st) > 0 else None,
            'questionnaire_id': questionnaire_id,
        }

    # MICI (MI-Inconsistent behaviors): severity global + harmful-behavior counts
    if qid == QuestionnaireID.MICI.value:
        globals_dict = data.get("globals", {})
        behaviors_dict = data.get("behaviors", {})
        if len(globals_dict) != len(MICI_GLOBAL_LABELS):
            raise ValueError(f"Expected {len(MICI_GLOBAL_LABELS)} globals, got {len(globals_dict)}")
        if len(behaviors_dict) != len(MICI_BEHAVIOR_LABELS):
            raise ValueError(f"Expected {len(MICI_BEHAVIOR_LABELS)} behaviors, got {len(behaviors_dict)}")
        global_scores = [globals_dict[label] for label in MICI_GLOBAL_LABELS]
        behavior_scores = [behaviors_dict[label] for label in MICI_BEHAVIOR_LABELS]
        scores_dict = {**globals_dict, **behaviors_dict}
        return {
            'scores_dict': scores_dict,
            'scores_list': global_scores + behavior_scores,
            'globals': globals_dict,
            'behaviors': behaviors_dict,
            'mean_score': sum(global_scores) / len(global_scores),
            'behavior_total': sum(behavior_scores),
            'questionnaire_id': questionnaire_id,
        }

    # MIPROC (MI process coder): one code per utterance for both speakers. Validate lengths
    # (when the caller knows the utterance counts) and enum membership; no numeric aggregate.
    if qid == QuestionnaireID.MIPROC.value:
        th = list(data.get("therapist_codes") or [])
        pt = list(data.get("patient_codes") or [])
        if expected_counts is not None:
            n_th, n_pt = expected_counts
            if len(th) != int(n_th):
                raise ValueError(f"Expected {int(n_th)} therapist codes, got {len(th)}")
            if len(pt) != int(n_pt):
                raise ValueError(f"Expected {int(n_pt)} patient codes, got {len(pt)}")
        bad_th = [c for c in th if c not in MIPROC_TH_CODES]
        if bad_th:
            raise ValueError(f"Unknown therapist code(s): {bad_th}")
        bad_pt = [c for c in pt if c not in MIPROC_PT_CODES]
        if bad_pt:
            raise ValueError(f"Unknown patient code(s): {bad_pt}")
        return {
            'scores_dict': {"MIPROC_TherapistCodes": th, "MIPROC_PatientCodes": pt},
            'therapist_codes': th,
            'patient_codes': pt,
            'questionnaire_id': questionnaire_id,
        }

    # Standard questionnaires with scores array
    scores = data.get("scores", [])
    if len(scores) != len(labels):
        raise ValueError(f"Expected {len(labels)} scores, got {len(scores)}")

    scores_dict = dict(zip(labels, scores, strict=False))

    return {
        'scores_dict': scores_dict,
        'scores_list': scores,
        'mean_score': sum(scores) / len(scores),
        'questionnaire_id': questionnaire_id,
    }
