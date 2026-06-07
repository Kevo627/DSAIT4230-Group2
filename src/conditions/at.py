from src.conditions.base import BaseCondition

PROMPT = """\
Given the image and the user's question below, generate ONE clarification question \
that you think is most appropriate to gain a better understanding of the user's intent.

The question contains referential ambiguity: the referring expression does not uniquely \
specify the intended referent — it is unclear which specific object, region, or entity \
in the image the user means.

The clarification question must ask the user to identify WHICH object or entity \
they are referring to — not ask for the answer to the original question itself.

Consider this ambiguity type when generating the clarification question.

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "clarification_question": "..."
}}"""


class ATCondition(BaseCondition):
    name = "at"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)