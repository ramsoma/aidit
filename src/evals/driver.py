#!/usr/bin/env python
"""driver.py

Orchestrates a doctor‑patient simulation loop by shuttling messages among the
three actors defined in *llm_wrappers.py*.

* Reads *initial patient prompts* from a Hugging Face dataset.
* Feeds each prompt to **Doctor**, obtains a reply, forwards that reply to
  **Patient**, and (optionally) asks **Judge** to grade the doctor's answer.
* Logs every turn and writes results to a JSONL file for later analysis.

This script focuses on glue logic; the heavy lifting (generation, evaluation)
resides in the wrapper classes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

from datasets import load_dataset  # type: ignore

from evals.doctor import Doctor
from evals.patient import Patient
from evals.judge import Judge


class SimulationDriver:
    """Coordinate message passing among Doctor, Patient, and Judge."""

    def __init__(
        self,
        doctor: Doctor,
        patient: Patient,
        judge: Judge | None = None,
        *,
        max_turns: int = 6,
    ) -> None:
        self.doctor = doctor
        self.patient = patient
        self.judge = judge
        self.max_turns = max_turns

    def run_dialogue(self, initial_prompt: str) -> Dict[str, Any]:
        """Run one multi‑turn interaction and return a structured log."""
        log: Dict[str, Any] = {
            "initial_prompt": initial_prompt,
            "turns": [],
        }

        current_utterance = initial_prompt
        for turn_idx in range(self.max_turns):
            # Doctor responds
            doctor_reply = self.doctor.generate_response(current_utterance)

            # Judge evaluates (if provided)
            evaluation: Dict[str, Any] | None = None
            if self.judge is not None:
                try:
                    evaluation = self.judge.evaluate_response(
                        conversation_history=self.patient.history,  # so far
                        doctor_response=doctor_reply,
                    )
                except NotImplementedError:
                    if turn_idx == 0:  # warn once
                        print("[WARN] Judge.evaluate_response not implemented – skipping scoring.")

            # Patient replies
            patient_reply = self.patient.generate_patient_response(doctor_reply)

            log["turns"].append(
                {
                    "turn": turn_idx,
                    "doctor": doctor_reply,
                    "patient": patient_reply,
                    "evaluation": evaluation,
                }
            )
            current_utterance = patient_reply

        return log


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:  # noqa: D401 – imperative style
    p = argparse.ArgumentParser(description="Run doctor‑patient simulation over a HF dataset")
    p.add_argument("dataset", help="Hugging Face dataset path (e.g. 'medical_dialogues')")
    p.add_argument("--split", default="train", help="Dataset split to use [default: train]")
    p.add_argument(
        "--text_field",
        default="text",
        help="Name of the column containing the initial patient utterance [default: text]",
    )
    p.add_argument("--model", required=True, help="Local HF model path or ID for Doctor")
    p.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    p.add_argument("--max_turns", type=int, default=6, help="Dialogue length [default: 6]")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("simulation_results.jsonl"),
        help="File to write JSONL logs [default: simulation_results.jsonl]",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------------
    # Instantiate actors
    # ---------------------------------------------------------------------

    doctor = Doctor(model_name_or_path=args.model)
    patient = Patient(openai_api_key=args.openai_key)
    judge = Judge(openai_api_key=args.openai_key)

    driver = SimulationDriver(doctor, patient, judge, max_turns=args.max_turns)

    # ---------------------------------------------------------------------
    # Load prompts
    # ---------------------------------------------------------------------

    ds = load_dataset(args.dataset, split=args.split)
    text_col = args.text_field
    if text_col not in ds.column_names:
        raise ValueError(f"Column '{text_col}' not found in {args.dataset} – available: {ds.column_names}")

    # ---------------------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------------------
    print(f"Running simulation on {len(ds)} entries…")

    with args.output.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(ds):
            initial_patient_prompt: str = example[text_col]
            log = driver.run_dialogue(initial_patient_prompt)
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
            if (idx + 1) % 10 == 0:
                print(f"…completed {idx + 1}/{len(ds)} dialogues")

    print(f"Done. Logs saved to {args.output}")


if __name__ == "__main__":
    main()
