from src.model import VLMWrapper, parse_json_output

SIMULATE_PROMPT = """\
You are a user who asked this question while looking at the image: "{ambiguous_question}"

A clarification question was posed to you: "{clarification_question}"

Select one of the options in the clarification question or point to which object you mean. \
Your response must identify ONLY which object or entity you are referring to. \
Do NOT answer the underlying question — only identify which object or option you mean.
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