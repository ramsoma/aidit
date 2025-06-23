"""
llm_wrappers.py

Defines lightweight wrapper classes around language models for a doctor–patient simulation workflow:

* **Doctor** – runs a local Hugging Face causal‑LM to draft medical answers.
* **Patient** – calls an OpenAI chat model to role‑play a patient given the running conversation.
* **Judge** – (placeholder) uses an OpenAI chat model to grade the doctor's answers.
"""

from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass, field

# External dependencies
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover – optional at import time
    openai = None  # Leave clear error downstream if used without install

__all__ = [
    "Doctor",
    "Patient",
    "Judge",
]

@dataclass
class Patient:
    """OpenAI‑based patient simulator that extends the running chat history."""

    openai_api_key: str
    system_prompt: str = (
        "You are a patient describing your symptoms in detail. Respond briefly and in first‑person,"\
        " focusing on your experience and feelings."
    )
    model: str = "gpt-4o-mini"
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if openai is None:
            raise ImportError("openai package required; `pip install openai`")
        openai.api_key = self.openai_api_key

    def generate_patient_response(self, doctor_utterance: str) -> str:
        """Append the doctor's last message and return the patient's next reply."""
        self.history.append({"role": "assistant", "content": doctor_utterance})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history
        response = openai.ChatCompletion.create(model=self.model, messages=messages)
        patient_message = response.choices[0].message.content
        # Keep conversation loop going
        self.history.append({"role": "user", "content": patient_message})
        return patient_message

    # Convenience alias
    __call__ = generate_patient_response
