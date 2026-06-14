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

Work through these steps and record your reasoning in the "reasoning" field:
1. List up to 3 plausible referents in the image that match the user's wording.
2. For each referent, state what the answer to the original question would be.
3. Identify the pair of referents whose answers differ most from each other.
4. Write a clarification question that directly separates that pair.
5. The final question should usually have the form:
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
  "reasoning": "...",
  "clarification_question": "..."
}}"""


class AnswerImpactCondition(BaseCondition):
    name = "answer_impact"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
