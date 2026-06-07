from src.conditions.base import BaseCondition

PROMPT = """\
Given the image and the user's question below, generate ONE clarification question \
that you think is most appropriate to gain a better understanding of the user's intent.

The clarification question must ask the user to identify WHICH object or entity \
they are referring to — not ask for the answer to the original question itself.

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "clarification_question": "..."
}}"""


class StandardCondition(BaseCondition):
    name = "standard"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)