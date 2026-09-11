"""Lazy local Qwen backend; importing this module does not load Torch or a model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence


class LLMError(RuntimeError):
    """Raised when the language-model backend cannot generate a response."""


class LLMBackend(Protocol):
    provider: str
    model_id: str

    def generate(self, messages: Sequence[dict[str, str]]) -> str: ...


class QwenTransformersLLM:
    """Run a public Qwen checkpoint locally through Hugging Face Transformers."""

    provider = "huggingface-transformers-local"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_id = str(config["model_id"])
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise LLMError(
                "Local Qwen dependencies are missing. Install requirements-llm.txt "
                "in a dedicated virtual environment."
            ) from exc
        if self.config.get("require_cuda", True) and not torch.cuda.is_available():
            raise LLMError(
                "CUDA is not available. Local Qwen inference is configured to require a GPU."
            )

        load_options: dict[str, Any] = {
            "device_map": self.config.get("device_map", "auto"),
            "torch_dtype": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": bool(self.config.get("trust_remote_code", False)),
            "local_files_only": bool(self.config.get("local_files_only", False)),
        }
        if self.config.get("quantization") == "bitsandbytes-4bit":
            compute_dtype_name = str(self.config.get("compute_dtype", "float16"))
            compute_dtype = getattr(torch, compute_dtype_name, None)
            if compute_dtype is None:
                raise LLMError(f"Unsupported Torch compute dtype: {compute_dtype_name}")
            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(
                    self.config.get("bnb_4bit_quant_type", "nf4")
                ),
                bnb_4bit_use_double_quant=bool(
                    self.config.get("bnb_4bit_use_double_quant", True)
                ),
                bnb_4bit_compute_dtype=compute_dtype,
            )
        offload_folder = self.config.get("offload_folder")
        if offload_folder:
            Path(offload_folder).mkdir(parents=True, exist_ok=True)
            load_options["offload_folder"] = offload_folder
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=load_options["trust_remote_code"],
            local_files_only=load_options["local_files_only"],
        )
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **load_options
            )
        except Exception as exc:
            raise LLMError(f"Could not load local Qwen model {self.model_id}: {exc}") from exc

    def generate(self, messages: Sequence[dict[str, str]]) -> str:
        self._load()
        tokenizer = self._tokenizer
        model = self._model
        try:
            prompt = tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(self.config.get("enable_thinking", False)),
            )
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_tokens = int(inputs["input_ids"].shape[-1])
            maximum = int(self.config.get("max_input_tokens", 12288))
            if input_tokens > maximum:
                raise LLMError(
                    f"Prompt has {input_tokens} tokens, above max_input_tokens={maximum}. "
                    "Reduce context.max_characters or use a larger context setting."
                )
            input_device = model.get_input_embeddings().weight.device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            generation_options: dict[str, Any] = {
                "max_new_tokens": int(self.config.get("max_new_tokens", 2048)),
                "do_sample": bool(self.config.get("do_sample", False)),
                "pad_token_id": tokenizer.eos_token_id,
            }
            if generation_options["do_sample"]:
                generation_options.update(
                    temperature=float(self.config.get("temperature", 0.7)),
                    top_p=float(self.config.get("top_p", 0.8)),
                    top_k=int(self.config.get("top_k", 20)),
                )
            outputs = model.generate(**inputs, **generation_options)
            generated = outputs[0, input_tokens:]
            return tokenizer.decode(generated, skip_special_tokens=True).strip()
        except LLMError:
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise LLMError(
                    "Qwen ran out of GPU memory. Lower context.max_characters or "
                    "model.max_new_tokens, close other GPU processes, then retry."
                ) from exc
            raise LLMError(f"Local Qwen generation failed: {exc}") from exc
