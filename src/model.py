import os
import json
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16
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
        if self.load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            print(f"Loading {self.model_name} in 4-bit on {self.device} ...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            dtype = _get_dtype(self.device)
            print(f"Loading {self.model_name} on {self.device} ({dtype}) ...")
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(self.device)
            
        """ split kaggle vs gpu"""
        dtype = _get_dtype(self.device)
        print(f"Loading {self.model_name} on {self.device} ({dtype}) ...")
        load_kwargs = {}
        if self.device == "cuda":
            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                device_map="auto",
            )
        else:
            load_kwargs.update(torch_dtype=dtype)

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        if self.device != "cuda":
            self._model = self._model.to(self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model.eval()
        print("Model ready.")

    @property
    def _input_device(self) -> str:
        # with device_map="auto" (4-bit), inputs must go to cuda explicitly
        return "cuda" if self.device == "cuda" else self.device
    def generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.2,
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

    def generate(self, image_path: str, prompt: str, max_new_tokens: int = 512,
                 do_sample: bool = False, temperature: float = 0.8, top_p: float = 0.95) -> str:
        self.load()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text", "text": prompt},
        ]}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self._input_device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def parse_json_output(text: str) -> dict:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return {"clarification_question": text.strip(), "_parse_failed": True}
