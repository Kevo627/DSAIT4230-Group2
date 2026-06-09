from src.conditions.base import BaseCondition


PROMPT = """\
The user's question contains referential ambiguity. Identify the single best
referential subtype from the list below, then generate ONE clarification question
that targets the missing referent.

Referential subtypes:
- Object identity: the object being referred to is unclear
- Person identity: the person being referred to is unclear
- Region/location: the relevant image region is unclear
- Text/sign identity: the relevant visible text or sign is unclear
- Attribute owner: the object or person that owns an attribute is unclear

Rules for the clarification question:
- Ask about the target referent only.
- Do NOT ask for the answer to the original question.
- Do NOT restate or paraphrase the original question.
- Ask a single, specific question.

User question:
{ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "referential_subtype": "...",
    "clarification_question": "..."
}}"""


class ATSubtypeCondition(BaseCondition):
    name = "at_subtype"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
