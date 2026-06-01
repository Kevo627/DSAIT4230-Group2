from src.conditions.base import BaseCondition

PROMPT = """Given the image and the user's visual question, generate a clarifying question to better understand the user's intent.

The question contains intent underspecification. Intent underspecification occurs when the user's question does not reveal which specific aspect or goal they care about, making it difficult to provide a precise answer.

The clarification question should confirm a specific interpretation of the user's goal, not ask the user to describe the image. Prefer a polar format: "Are you asking about X?" For example: "Are you asking about the architectural style of the building, or its historical background?"

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "clarification_question": "..."
}}"""


class ATCondition(BaseCondition):
    name = "at"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)