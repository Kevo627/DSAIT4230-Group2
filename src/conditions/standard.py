from src.conditions.base import BaseCondition

PROMPT = """\
Given the image and the user's question below, generate ONE clarification question \
that would help answer the user more accurately. Target the most important missing \
information or ambiguity.

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "clarification_question": "..."
}}"""


class StandardCondition(BaseCondition):
    name = "standard"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)