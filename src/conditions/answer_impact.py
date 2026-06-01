from src.conditions.base import BaseCondition

PROMPT = """\
The user's visual question is intent-underspecified: the image may contain enough
visual evidence, but the user's intended information need is unclear.

Generate the clarification question that would most improve the final answer.
Do this by reasoning about answer impact, not just by naming the ambiguity type.

Steps:
1. Identify 3 to 5 plausible user intents for the ambiguous question.
2. For each intent, state what kind of final answer would likely be needed.
3. Decide which intents would lead to meaningfully different answers.
4. Ask ONE concise clarification question that best separates the answer-relevant
   intents.

Rules:
- Ask about the user's intended goal, aspect, decision criterion, context, or
  desired output.
- Do NOT ask for the answer to the original question.
- Do NOT ask a generic question such as "Can you clarify?"
- Prefer a question whose answer would change or sharpen the final response.
- Ground the question in the image and original question when possible.

User question: {ambiguous_question}

Return ONLY a JSON object, no extra text:
{{
    "plausible_intents": [
        {{
            "intent": "...",
            "expected_answer_type": "...",
            "would_change_answer": true
        }}
    ],
    "answer_impact": "...",
    "reasoning": "...",
    "clarification_question": "..."
}}"""


class AnswerImpactCondition(BaseCondition):
    name = "answer_impact"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
