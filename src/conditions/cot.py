from src.conditions.base import BaseCondition

PROMPT = """\
Given the image and the user's question below, first briefly reason about why the \
question is ambiguous and what information is missing. Then generate ONE clarification question based on your reasoning.
The clarification question must ask the user what they want to know, not what is visible in the image. The user is asking you to describe the image — ask them which specific aspect or goal they have in mind.

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