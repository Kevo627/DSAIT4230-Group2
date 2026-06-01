from src.conditions.base import BaseCondition

PROMPT = """The user's question is ambiguous because their intended goal is unclear.

Given the image and the user's question below, generate ONE clarification question that asks the user to specify which aspect or goal they have in mind.

The user is asking YOU to describe the image — they cannot see it themselves. Do NOT ask them what is visible. Instead, ask them to clarify which specific aspect or goal they have in mind.

BAD (asks user to observe image): "What safety gear is visible on the skateboarder?"
GOOD (asks user to clarify intent): "Are you asking about a specific type of gear, or whether the skateboarder has any protection at all?"

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "clarification_question": "..."
}}"""


class StandardCondition(BaseCondition):
    name = "standard"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)