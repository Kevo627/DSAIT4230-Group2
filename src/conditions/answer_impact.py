from src.conditions.base import BaseCondition


# Difference from Standard / AT-CoT:
#   - Standard directly generates one referential clarification question.
#   - AT-CoT reasons about the referential ambiguity before asking.
#   - This strategy first identifies several plausible referents, then asks one
#     clarification question designed to separate the referents that would most
#     change the final answer.

PROMPT = """\
You are given an image and an ambiguous visual question.

The ambiguity is REFERENTIAL: the user's wording does not uniquely identify
which visible object, region, person, text, or entity they mean.

Your task is to ask ONE clarification question about the target referent.

Important distinction:
- Do NOT answer the visual question.
- Do NOT ask the user to verify a final answer.
- DO ask which visible referent the user means.

User question:
{ambiguous_question}

Follow these steps internally:
1. Identify up to 3 plausible referents in the image that could match the user's wording.
2. Consider whether the final answer would change depending on which referent is meant.
3. Choose the clarification question that best separates the answer-relevant referents.
4. The final question should usually have the form:
   "Are you referring to X, Y, or Z?"
   or
   "Do you mean X or Y?"

Rules:
- The clarification question must ask about the target referent.
- Do NOT ask for the final answer.
- Do NOT ask generic questions like "Can you clarify?"
- Do NOT simply restate the original question.
- Prefer concrete visible options when possible.
- Keep the final clarification question concise and directly answerable.

Return ONLY this JSON object, no extra text:
{{
  "clarification_question": "..."
}}"""


class AnswerImpactCondition(BaseCondition):
    name = "answer_impact"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
