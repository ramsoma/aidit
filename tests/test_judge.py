import unittest
from unittest.mock import patch, MagicMock
from src.evals.judge import Judge

class TestJudge(unittest.TestCase):
    def setUp(self):
        self.judge = Judge(openai_api_key="sk-test")
        self.conversation_history = [
            {"role": "user", "content": "I have a headache and fever."},
            {"role": "assistant", "content": "<think>Possible causes: flu, cold, COVID-19.</think>How long have you had these symptoms?"},
            {"role": "user", "content": "About 2 days."},
        ]
        self.doctor_response = "<think>Flu is most likely, but COVID-19 is possible.</think>Do you have a cough?"

    @patch("src.evals.judge.openai.ChatCompletion.create")
    def test_judge_parses_json_response(self, mock_create):
        # Simulate OpenAI returning a valid JSON string
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"info_gathering_score": 8, "diagnosis_score": 9, "member_feedback_score": 10, "feedback": "Good job."}'))]
        )
        result = self.judge.evaluate_response(self.conversation_history, self.doctor_response)
        self.assertEqual(result["info_gathering_score"], 8)
        self.assertEqual(result["diagnosis_score"], 9)
        self.assertEqual(result["member_feedback_score"], 10)
        self.assertIn("Good job", result["feedback"])

    @patch("src.evals.judge.openai.ChatCompletion.create")
    def test_judge_fallback_on_non_json(self, mock_create):
        # Simulate OpenAI returning non-JSON text
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Sorry, I cannot evaluate this.'))]
        )
        result = self.judge.evaluate_response(self.conversation_history, self.doctor_response)
        self.assertIsNone(result["info_gathering_score"])
        self.assertIsNone(result["diagnosis_score"])
        self.assertIsNone(result["member_feedback_score"])
        self.assertIn("Sorry", result["feedback"])

    @patch("src.evals.judge.openai.ChatCompletion.create")
    def test_prompt_contains_expected_sections(self, mock_create):
        # Check that the prompt sent to OpenAI contains the right sections
        def side_effect(*args, **kwargs):
            messages = kwargs["messages"]
            user_prompt = messages[1]["content"]
            self.assertIn("Conversation so far:", user_prompt)
            self.assertIn("Doctor's latest response:", user_prompt)
            # Return valid JSON
            return MagicMock(choices=[MagicMock(message=MagicMock(content='{"info_gathering_score": 7, "diagnosis_score": 7, "member_feedback_score": 7, "feedback": "OK"}'))])
        mock_create.side_effect = side_effect
        result = self.judge.evaluate_response(self.conversation_history, self.doctor_response)
        self.assertEqual(result["info_gathering_score"], 7)

if __name__ == "__main__":
    unittest.main() 