from src.model import VLMWrapper, parse_json_output

SIMULATE_PROMPT = """\
You are simulating the response of a user who asked an ambiguous question while \
looking at this image.

The user's original (ambiguous) question was: "{ambiguous_question}"
The user's actual intent was:               "{gold_intended_question}"

A clarification question was posed to the user: "{clarification_question}"

Answer the clarification question as this user would — in 1–2 short, natural sentences. \
Be specific about which object or region in the image you mean, consistent with the \
user's actual intent. Do not reveal the intended question directly.

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
        gold_intended_question: str,
        clarification_question: str,
    ) -> dict:
        prompt = SIMULATE_PROMPT.format(
            ambiguous_question=ambiguous_question,
            gold_intended_question=gold_intended_question,
            clarification_question=clarification_question,
        )
        raw = self.model.generate(image_path, prompt, do_sample=False)
        parsed = parse_json_output(raw)
        return {
            "user_response": parsed.get("user_response", raw.strip()),
            "raw_output": raw,
            "_parse_failed": parsed.get("_parse_failed", False),
        }
