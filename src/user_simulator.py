from src.model import VLMWrapper, parse_json_output

SIMULATE_PROMPT = """\
You are a user who asked this question while looking at the image: "{ambiguous_question}"
Your actual intent was: "{gold_intended_question}"

A clarification question was posed to you: "{clarification_question}"

Using your intent to identify which object or option you mean, respond to the \
clarification question naturally. Identify ONLY which object or entity you are \
referring to — do not answer the underlying question itself. \
Do NOT mention any object, person, or detail that is not clearly visible in the image \
or not present in the clarification question options. \
Keep it to 1 short sentence.

Template examples of correct responses:
- CQ: "Are you referring to <object A> or <object B>?" → "The <object A>, yes."
- CQ: "Do you mean <object A> or <object B>?" → "The <object B> on the left."
- CQ: "Are you asking about <option A> or <option B>?" → "The <option A>, the one near the <landmark>."
- CQ: "Do you mean <object>?" → "Yes, exactly that one."

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
        for _ in range(3):
            raw = self.model.generate(image_path, prompt, do_sample=False)
            parsed = parse_json_output(raw)
            if not parsed.get("_parse_failed") and "user_response" in parsed:
                break
        return {
            "user_response": parsed.get("user_response", raw.strip()),
            "raw_output": raw,
            "_parse_failed": parsed.get("_parse_failed", False),
        }