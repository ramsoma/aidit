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
    """Medical judge that grades doctor answers for information gathering, diagnosis, and member feedback following."""

    openai_api_key: str
    evaluation_prompt: str = (
        "You are a strict medical examiner. Assess the assistant's clinical performance in three areas: "
        "(a) Information gathering: Did the assistant collect sufficient and relevant information before making a diagnosis? "
        "(b) Differential diagnosis accuracy: How accurate and reasonable is the differential diagnosis in any <think>...</think> tokens? "
        "(c) Member feedback following: Did the assistant respect the user's (patient's) wishes, e.g., not asking further questions when the user refused? "
        "Reply in JSON with keys: info_gathering_score (1-10), diagnosis_score (1-10), member_feedback_score (1-10), and feedback (string)."
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
        """Return a dict with sub-scores and feedback for information gathering, diagnosis, and member feedback following."""
        # Compose a detailed prompt for the LLM
        history_str = "\n".join([
            f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history
        ])
        prompt = (
            f"{self.evaluation_prompt}\n\n"
            f"Conversation so far:\n{history_str}\n"
            f"\nDoctor's latest response:\n{doctor_response}\n"
            f"\nEvaluate the above."
        )
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": self.evaluation_prompt},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        # Parse JSON from the model's response
        import json
        import re
        content = response.choices[0].message.content
        # Extract JSON from the response (robust to extra text)
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        # Fallback: return raw content as feedback
        return {
            "info_gathering_score": None,
            "diagnosis_score": None,
            "member_feedback_score": None,
            "feedback": content.strip(),
        }

    __call__ = evaluate_response
