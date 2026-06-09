from src.conditions.base import BaseCondition


PROMPT = """\
Given the image and the user's question below, generate ONE clarification question
that resolves referential ambiguity.

Before generating the clarification question, explain why the referring
expression is ambiguous and which plausible referents could match it.

The clarification question must ask the user to identify the target referent,
not ask for the answer to the original question.

User question:
{ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "reasoning": "...",
    "clarification_question": "..."
}}"""


class CoTCondition(BaseCondition):
    name = "cot"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
