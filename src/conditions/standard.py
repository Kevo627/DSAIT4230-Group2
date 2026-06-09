from src.conditions.base import BaseCondition

PROMPT = """\
### Instruction:
You are given an image and an ambiguous visual question.

Your task is NOT to answer the question.
Your task is ONLY to ask one clarification question that would help determine what the user wants to know.

The input question is known to be ambiguous due to missing user intent.
Ask about the user's intended aspect, goal, criterion, or desired answer type.

Do not:
- answer the visual question
- explain the ambiguity
- list possible answers
- identify objects in the image
- rewrite the original question

Your clarification question should usually start with:
- "Are you asking about..."
- "Do you want to know..."
- "Which aspect..."

### Question:
{ambiguous_question}

### Output:
Return only this JSON object:
{{
  "clarification_question": "..."
}}"""


class StandardCondition(BaseCondition):
    name = "standard"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)