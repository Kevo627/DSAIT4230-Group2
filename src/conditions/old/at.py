from src.conditions.base import BaseCondition


PROMPT = """\
Given the image and the user's question below, generate ONE clarification question
that resolves referential ambiguity.

The question contains referential ambiguity: the referring expression does not
uniquely specify the target referent. It is unclear which specific object,
region, person, text, or entity in the image the user means.

The clarification question must ask the user to identify WHICH referent they are
asking about, not ask for the answer to the original question. If multiple
plausible referents are visible, offer them as specific options. If only one
plausible referent exists, ask "Do you mean X?"

User question:
{ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "clarification_question": "..."
}}"""


class ATCondition(BaseCondition):
    name = "at"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
