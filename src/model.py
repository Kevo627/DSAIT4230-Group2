"""
VLM wrapper for Qwen2.5-VL on Apple Silicon (MPS).

Lazy-loads the model on first call so importing this module is free.
"""

import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class VLMWrapper:
    def __init__(self, model_name: str = DEFAULT_MODEL, load_in_4bit: bool = False):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = _get_device()
        self._model = None
        self._processor = None

    def load(self):
        if self._model is not None:
            return
        dtype = _get_dtype(self.device)
        print(f"Loading {self.model_name} on {self.device} ({dtype}){' [4-bit]' if self.load_in_4bit else ''} ...")
        kwargs = {"torch_dtype": dtype}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = self.device
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model.eval()
        print("Model ready.")

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if not self.load_in_4bit:
            inputs = inputs.to(self.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def parse_json_output(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            return obj
        except json.JSONDecodeError:
            pass

    return {"clarification_question": text.strip(), "_parse_failed": True}