from src.conditions.base import BaseCondition

PROMPT = """\
Given the image and the user's question below, generate ONE clarification question \
that you think is most appropriate to gain a better understanding of the user's intent.

Before generating the clarification question, provide a textual explanation of your \
reasoning about why the question is ambiguous and how you plan to clarify it.

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "reasoning": "...",
    "clarification_question": "..."
}}"""


class CoTCondition(BaseCondition):
    name = "cot"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)