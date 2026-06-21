from src.conditions.base import BaseCondition


PROMPT = """\
Given the image and the user's question below, generate ONE clarification question
that would help you better understand what the user is asking.

The clarification question should target the most important missing information
that would allow you to give the most accurate and helpful answer.

Do not:
- answer the visual question
- list possible final answers
- rewrite the original question

Your clarification question should usually start with:
- "Do you mean..."
- "Which one..."

User question:
{ambiguous_question}

Return only this JSON object:
{{
  "clarification_question": "..."
}}"""


class StandardCondition(BaseCondition):
    name = "standard"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
