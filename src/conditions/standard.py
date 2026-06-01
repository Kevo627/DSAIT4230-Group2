from src.conditions.base import BaseCondition

PROMPT = """\
Given the image and the user's question below, generate ONE clarification question \
that would help answer the user more accurately. Target the most important missing \
information or ambiguity. The clarification question must ask the user what they want to know, not what is visible in the image. The user is asking you to describe the image — ask them which specific aspect or goal they have in mind.

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "clarification_question": "..."
}}"""


class StandardCondition(BaseCondition):
    name = "standard"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)