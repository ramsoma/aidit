from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass, field

try:
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except ImportError:  # pragma: no cover – optional at import time
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    torch = None  # type: ignore

class Doctor:
    """Simple wrapper around a *local* Hugging Face causal‑LM.

    Parameters
    ----------
    model_name_or_path : str
        Path or HF hub ID for the model.
    device : str, default "cpu"
        Torch device to load the model onto.
    tokenizer_name_or_path : str | None, default None
        Optional separate tokenizer path/ID (falls back to *model_name_or_path*).
    max_new_tokens : int, default 256
        Maximum length for generated response.
    temperature : float, default 0.8
        Sampling temperature.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        tokenizer_name_or_path: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ) -> None:
        if AutoModelForCausalLM is None:
            raise ImportError(
                "transformers must be installed to use Doctor; `pip install transformers`"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name_or_path)
            .to(device)
            .eval()
        )
        self.device = device
        self.generation_cfg = dict(max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)

    def generate_response(self, utterance: str, **gen_kwargs: Any) -> str:
        """Generate a reply to *utterance* and return *only* the newly generated text."""
        if torch is None:
            raise RuntimeError("torch is required to run the local model")

        with torch.inference_mode():
            inputs = self.tokenizer(utterance, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(**inputs, **self.generation_cfg, **gen_kwargs)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Heuristic: strip the prompt off the front
        return decoded[len(utterance):].strip()

    # Allow the instance to be called directly (syntactic sugar)
    __call__ = generate_response

