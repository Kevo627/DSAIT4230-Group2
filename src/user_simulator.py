from src.model import VLMWrapper, parse_json_output

SIMULATE_PROMPT = """\
You are a user who asked this question while looking at the image: "{ambiguous_question}"

A clarification question was posed to you: "{clarification_question}"

Answer the clarification question naturally and concisely, in 1 short sentence, \
based on what you see in the image.

Return ONLY a JSON object, no extra text:
{{
    "user_response": "..."
}}"""

class UserSimulator:
    def __init__(self, model: VLMWrapper):
        self.model = model

    def simulate(
        self,
        image_path: str,
        ambiguous_question: str,
        clarification_question: str,
    ) -> dict:
        prompt = SIMULATE_PROMPT.format(
            ambiguous_question=ambiguous_question,
            clarification_question=clarification_question,
        )
        raw = self.model.generate(image_path, prompt, do_sample=False)
        parsed = parse_json_output(raw)
        return {
            "user_response": parsed.get("user_response", raw.strip()),
            "raw_output": raw,
            "_parse_failed": parsed.get("_parse_failed", False),
        }