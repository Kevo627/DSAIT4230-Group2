from src.conditions.base import BaseCondition

PROMPT = """\
### Instruction:
You are a clarification-question generator for ambiguous visual questions.

Your task is NOT to answer the user's visual question.
Your task is ONLY to ask one clarification question.

The user's question is ambiguous because the user's underlying intent is unclear:
their goal, desired aspect, success criterion, context, or expected answer type
could have multiple valid interpretations.

Think step by step, but do not resolve the ambiguity yourself:
1. Identify what part of the user's intent is underspecified.
2. List 2 or 3 plausible user intents that could fit the question and image.
3. State what intent information is missing from the user's question.
4. Generate ONE clarification question that asks the user to specify that
   missing information.

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
