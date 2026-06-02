from src.conditions.base import BaseCondition




# Difference from Standard / AT / CoT / AT-CoT:
#   - Standard directly generates one clarification question.
#   - AT uses ambiguity-type information.
#   - CoT reasons before generating one question.
#   - AT-CoT reasons using ambiguity type before generating one question.
#   - This strategy first generates several plausible user intents, then asks
#     one clarification question designed to separate the most answer-relevant
#     intents.

# Possible related literature:
# Uncertainty of Thoughts (Huang et al. 2024) - reasoning about multiple possible user intents, but not focused on visual questions or ambiguity types.
# Clarifying Ambiguities: on the Role of Ambiguity Types in Prompting Methods for Clarification Generation (Tang et al. 2025) - the core ambiguity reasoning that we already use in AT-COT


PROMPT = """
You are given an image and an ambiguous visual question.

The ambiguity is INTENT UNDERSPECIFICATION:
the user may be referring to a visible target, but it is unclear what aspect, property, purpose, meaning, or evaluation they want to know.

Your task is to ask ONE clarification question about the user's intended information need.

Important distinction:
- Do NOT ask the user to answer the visual question.
- Do NOT ask the user to verify a fact visible in the image.
- DO ask what aspect, property, interpretation, criterion, or context the user wants the model to focus on.

User question:
{ambiguous_question}

Follow these steps internally:
1. Identify 3 plausible interpretations of what the user might be asking about.
2. Rewrite these interpretations as aspects or intents, not as possible answers.
3. Choose the clarification question that best separates these possible intents.
4. The final question should usually have the form:
   "Are you asking about X, Y, or Z?"
   or
   "Do you want to know about X or Y?"

Rules:
- The clarification question must ask about the user's intent, not about the image content itself.
- Do NOT ask yes/no questions about whether something is present in the image.
- Do NOT ask for the final answer.
- Do NOT ask generic questions like "Can you clarify?"
- Do NOT simply restate the original question.
- Prefer questions that offer 2 or 3 possible aspects, properties, or interpretations.
- Keep the final clarification question concise and directly answerable.

Bad examples:
- "Is the skateboarder wearing protective gear?"
- "Is the siding made of wood?"
- "Is the sofa covered for protection?"

Good examples:
- "Are you asking about the skateboarder's clothing, protective equipment, or whether the outfit is safe?"
- "Are you asking about the siding's material, condition, or purpose?"
- "Are you asking why the sofa is covered, what it is covered with, or whether the covering is decorative?"

Return ONLY this JSON object, no extra text:
{{
  "reasoning": "The underspecified intent is ... Plausible user intents include ... The missing intent information is ...",
  "clarification_question": "..."
}}"""


class AnswerImpactCondition(BaseCondition):
    name = "answer_impact"

    def build_prompt(self, ambiguous_question: str) -> str:
        return PROMPT.format(ambiguous_question=ambiguous_question)
