from src.conditions.base import BaseCondition


PROMPT = """\
Given the image and the user's question below, generate ONE clarification question
that best resolves the referential ambiguity in the user's question.

The question contains referential ambiguity: the referring expression does not
uniquely specify the target referent. It is unclear which specific object,
region, person, text, or entity in the image the user means.

Work through the following steps, writing your reasoning in the "reasoning" field:
1. Why the referring expression is ambiguous in this image.
2. Which plausible referents could match the user's wording.
3. Which clarification question would separate those referents most directly.

Rules:
- Do NOT answer the original visual question.
- Do NOT state which referent is correct.
- Do NOT ask the user for the final answer.
- Ask the user to identify the target referent.
- If multiple plausible referents are visible, offer them as concrete options.
- If only one plausible referent exists, ask "Do you mean X?"

User question:
{ambiguous_question}

Return ONLY this JSON object, no extra text:
{{
  "reasoning": "...",
  "clarification_question": "..."
}}"""


class ATCoTCondition(BaseCondition):
    name = "at_cot"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
