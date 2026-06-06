from src.model import VLMWrapper, parse_json_output

SIMULATE_PROMPT = """\
You are simulating a user who asked this question while looking at the image: "{ambiguous_question}"

A clarification question was posed to you: "{clarification_question}"

Answer only what the clarification question asks. Be specific about which object or region \
in the image you mean. Keep it to 1-2 natural sentences.

Note: the user's actual intended referent is: "{gold_intended_question}" — use this only \
to identify which object in the image to refer to, not to add information beyond what the \
clarification question asks for.

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
