from src.conditions.base import BaseCondition

PROMPT = """\
Given the image and the user's question below, generate ONE clarification question \
that you think is most appropriate to gain a better understanding of the user's intent.

The question contains referential ambiguity: the referring expression does not uniquely \
specify the intended referent — it is unclear which specific object, region, or entity \
in the image the user means.

The clarification question must ask the user to identify WHICH object or entity \
they are referring to — not ask for the answer to the original question itself. \
If multiple plausible referents are visible, offer them as specific options based \
on what is in the image. If only one plausible referent exists, ask \"Do you mean X?\" \
Do not ask the user to define terms or re-identify what they already said.

Before generating the clarification question, provide a textual explanation of your \
reasoning about why this referential ambiguity exists in the image context — identifying \
the plausible referents — and how you plan to clarify it.

Hard rules:
- Do NOT answer the original visual question.
- Do NOT state which interpretation is correct.
- Do NOT say "it is", "there is", "the answer is", or give a final answer.
- Do NOT ask the user to identify a visible object unless that is necessary to
  clarify their intent.
- The clarification_question field MUST be a question addressed to the user.
- The clarification_question should usually start with "Are you asking about",
  "Do you want to know", "Which aspect", or "What specific".

User question: {ambiguous_question}

Return ONLY this JSON object, no extra text:
{{
  "reasoning": "The underspecified intent is ... Plausible user intents include ... The missing intent information is ...",
  "clarification_question": "..."
}}"""


class ATCoTCondition(BaseCondition):
    name = "at_cot"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
