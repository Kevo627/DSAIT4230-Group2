from src.conditions.base import BaseCondition


PROMPT = """\
Given the image and the user's question below, generate ONE clarification question
that best resolves the referential ambiguity in the user's question.

The clarification question must help identify which object, region, person, text,
or entity the user is referring to. Do not ask for the answer to the original
question itself.

If multiple plausible referents are visible, offer them as specific options. If
only one plausible referent is likely, ask whether the user means that referent.

Do not:
- answer the visual question
- explain the ambiguity
- list possible final answers
- rewrite the original question

Your clarification question should usually start with:
- "Are you referring to..."
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
