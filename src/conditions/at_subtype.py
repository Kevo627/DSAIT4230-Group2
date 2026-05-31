from src.conditions.base import BaseCondition

PROMPT = """\
The user's question contains intent underspecification. Identify the single best
ambiguity subtype from the list below, then generate ONE clarification question
that targets the missing information for that subtype.

Ambiguity subtypes:
- Answer-type / granularity: the expected level or type of answer is unclear
- Aspect: the relevant property or aspect is unspecified
- Decision criterion: a choice is requested but the basis for choosing is missing
- Context / relation: the relevant relation, cause, role, or situation frame is unclear
- Output format: the desired response format is unspecified

Rules for the clarification question:
- Ask about missing intent only (goal, constraints, output format, decision criteria, or context).
- Do NOT ask for the answer to the original question.
- Do NOT restate or paraphrase the original question.
- Ask a single, specific question.

Bad: "What is in the image?"
Bad: "Can you answer the question?"
Good: "Which part of the image should the answer focus on (object, text, or layout)?"
Good: "What output format do you want (short sentence, bullet list, or JSON)?"

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "ambiguity_subtype": "...",
    "clarification_question": "..."
}}"""


class ATSubtypeCondition(BaseCondition):
    name = "at_subtype"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
