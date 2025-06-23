from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass, field

# External dependencies
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover – optional at import time
    openai = None  # Leave clear error downstream if used without install

@dataclass
class Judge:
    """Placeholder wrapper that will grade doctor answers via OpenAI.

    The *evaluate_response* method is intentionally left incomplete. Plug in your own
    evaluation rubric or LM‑scoring logic.
    """

    openai_api_key: str
    evaluation_prompt: str = (
        "You are a strict medical examiner. Assess the assistant's\n"
        "clinical accuracy, clarity, empathy, and safety. Reply in JSON with keys:\n"
        "score (1–10) and feedback (string)."
    )
    model: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        if openai is None:
            raise ImportError("openai package required; `pip install openai`")
        openai.api_key = self.openai_api_key

    def evaluate_response(
        self,
        conversation_history: List[Dict[str, str]],
        doctor_response: str,
    ) -> Dict[str, Any]:
        """Return a dict with *score* and *feedback* — *not yet implemented*."""
        # TODO: Implement actual OpenAI call and JSON parsing.
        raise NotImplementedError("evaluate_response is not implemented; fill in your logic here.")

    __call__ = evaluate_response
