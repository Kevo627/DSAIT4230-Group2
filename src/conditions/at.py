from src.conditions.base import BaseCondition

PROMPT = """\
The user's question contains intent underspecification — the user's underlying goal, \
desired output, success criteria, or expected format are unclear or open to multiple \
interpretations.

Given the image and the user's question below, generate ONE clarification question targeting the missing intent.
The clarification question must ask the user what they want to know, not what is visible in the image. The user is asking you to describe the image — ask them which specific aspect or goal they have in mind.

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "clarification_question": "..."
}}"""


class ATCondition(BaseCondition):
    name = "at"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)