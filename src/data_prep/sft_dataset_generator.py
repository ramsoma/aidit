import os
import openai
from datasets import Dataset
from tqdm import tqdm
import logging
import random
import time
import pandas as pd
import argparse
from typing import List, Dict, Any, Tuple

# --- Configuration ---
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY" # Environment variable to look for
# Default Base URL for Google's Generative Language API (OpenAI compatible)
OPENAI_API_BASE_URL_DEFAULT = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Default model name changed to a Gemini model
LLM_MODEL_NAME_DEFAULT = "models/gemini-2.0-flash" # Example Gemini model

# Default system prompt for the start of each conversation transcript
SYSTEM_PROMPT_CONVERSATION_DEFAULT = "You are a helpful medical assistant."

# Target dataset size
TARGET_MIN_EXAMPLES_DEFAULT = 100
TARGET_MAX_EXAMPLES_DEFAULT = 500

# Text for the 10th question check-in
TENTH_QUESTION_CHECK_IN_TEXT = (
    "We've covered a fair bit of ground. Are you comfortable answering a few more questions, "
    "or would you prefer I provide an assessment based on what we've discussed so far?"
)

# Keywords to heuristically detect user's desire for assessment after check-in
USER_WANTS_ASSESSMENT_KEYWORDS = ["assessment", "diagnosis", "tell me", "what do you think", "yes please", "provide it", "current information"]
USER_DECLINES_MORE_QUESTIONS_KEYWORDS = ["no more questions", "no thanks", "stop", "don't want more", "no more q"]

# New: Minimum questions before LLM should consider if diagnosis is narrowed enough
MIN_QUESTIONS_FOR_CONSIDERING_EARLY_DIAGNOSIS = 5


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SFTDatasetGenerator:
    def __init__(self, api_key: str, model_name: str = LLM_MODEL_NAME_DEFAULT, base_url: str = OPENAI_API_BASE_URL_DEFAULT):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.client = None

        if self.api_key and self.api_key != "YOUR_OPENAI_API_KEY_HERE": # Placeholder check
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url # Use the specified base_url
                )
                logging.info(f"OpenAI client initialized for model {self.model_name} at base URL {self.base_url}.")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
        else:
            logging.error("OpenAI API key not provided or is a placeholder. LLM calls will be skipped.")

    def _call_openai_llm_for_label(self, prompt_for_label_generation: str, max_retries: int = 3, initial_delay: int = 1) -> str:
        """
        Calls the LLM (now potentially Gemini via OpenAI-compatible endpoint)
        to generate the full label (think + answer) based on a constructed prompt.
        """
        if not self.client:
            logging.warning("OpenAI client not initialized. Skipping LLM call for label generation.")
            # Return a structured error message within the expected label format
            return "<think>\nDifferential diagnosis: LLM_CALL_SKIPPED_NO_CLIENT\nRationale for next action: Client not available.\nQuestion count for assistant: N/A\n</think>\nError: LLM client not available."

        messages_for_api = [{"role": "user", "content": prompt_for_label_generation}]
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages_for_api,
                    temperature=0.5, 
                    max_tokens=1024  # Increased slightly more for complex decision + diagnosis
                )
                content = response.choices[0].message.content
                if content is None:
                    logging.warning("LLM returned None content.")
                    logging.info(response)
                    continue
                return content.strip() 
            except openai.RateLimitError as e:
                logging.warning(f"API RateLimitError (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2 
            except Exception as e: 
                logging.error(f"API call for label failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return f"<think>\nDifferential diagnosis: LLM_CALL_FAILED_NON_RATELIMIT\nRationale for next action: LLM call failed.\nQuestion count for assistant: N/A\n</think>\nError: LLM call failed after retries due to: {str(e)[:100]}..."
                time.sleep(delay)
                delay *= 2
        return f"<think>\nDifferential diagnosis: LLM_CALL_FAILED_ALL_RETRIES\nRationale for next action: LLM call failed.\nQuestion count for assistant: N/A\n</think>\nError: LLM call failed after all retries."

    def load_conversations_from_csv(self, file_path: str) -> List[List[Dict[str, str]]]:
        source_conversations: List[List[Dict[str, str]]] = []
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            logging.error(f"CSV file not found at path: {file_path}")
            return source_conversations
        except Exception as e:
            logging.error(f"Error reading CSV file {file_path}: {e}")
            return source_conversations

        required_columns = ['conversation_id', 'doctor_response', 'patient_response']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"CSV file must contain columns: {', '.join(required_columns)}. Found: {', '.join(df.columns)}")
            return source_conversations

        grouped = df.groupby('conversation_id')
        logging.info(f"Found {len(grouped)} unique conversation_ids in {file_path}.")
        
        for conv_id, group in tqdm(grouped, desc="Loading Conversations from CSV"):
            conv_history_for_one_id: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_CONVERSATION_DEFAULT}]
            group = group.sort_index() 

            for _, row in group.iterrows():
                if pd.notna(row['doctor_response']) and str(row['doctor_response']).strip():
                    conv_history_for_one_id.append({"role": "assistant", "content": str(row['doctor_response'])})
                if pd.notna(row['patient_response']) and str(row['patient_response']).strip():
                    conv_history_for_one_id.append({"role": "user", "content": str(row['patient_response'])})
            
            if len(conv_history_for_one_id) > 1: 
                source_conversations.append(conv_history_for_one_id)
            else:
                logging.warning(f"Conversation ID {conv_id} resulted in an empty history. Skipping.")
                
        logging.info(f"Successfully loaded {len(source_conversations)} valid conversations from {file_path}")
        return source_conversations

    def _format_context_for_label_prompt(self, context_messages: List[Dict[str, str]]) -> str:
        formatted_str = ""
        for message in context_messages:
            if message["role"] == "system": 
                continue
            role_display = "Patient" if message["role"] == "user" else "Assistant"
            formatted_str += f"{role_display}: {message['content']}\n\n"
        return formatted_str.strip()

    def generate_sft_label(self, context_for_label_generation: List[Dict[str, str]], assistant_question_count: int) -> str:
        formatted_history = self._format_context_for_label_prompt(context_for_label_generation)
        
        action_type = "ask_follow_up" # Default action
        rationale_for_action = "[LLM to provide reasoning for the question it will ask]" 
        answer_instruction = "Your response should be a single, pertinent follow-up question to the patient, aimed at gathering more specific information or clarifying symptoms."

        # Scenario 1: User explicitly wants diagnosis after check-in
        if len(context_for_label_generation) >= 2:
            last_user_turn = context_for_label_generation[-1]
            second_last_turn = context_for_label_generation[-2]
            if last_user_turn["role"] == "user" and \
               second_last_turn["role"] == "assistant" and \
               second_last_turn["content"] == TENTH_QUESTION_CHECK_IN_TEXT:
                user_response_content = last_user_turn["content"].lower()
                if any(keyword in user_response_content for keyword in USER_WANTS_ASSESSMENT_KEYWORDS) or \
                   any(keyword in user_response_content for keyword in USER_DECLINES_MORE_QUESTIONS_KEYWORDS):
                    action_type = "provide_diagnosis_user_request"

        # Scenario 2: Time for 10th question check-in (if not overridden by user request for diagnosis)
        if action_type == "ask_follow_up" and \
           assistant_question_count > 0 and assistant_question_count % 10 == 0:
            action_type = "ask_10th_question_check_in"

        # Scenario 3: Consider early diagnosis if enough questions asked (and not other scenarios)
        if action_type == "ask_follow_up" and \
           assistant_question_count >= MIN_QUESTIONS_FOR_CONSIDERING_EARLY_DIAGNOSIS:
            action_type = "consider_early_diagnosis"


        # --- Set instructions based on determined action_type ---
        if action_type == "provide_diagnosis_user_request":
            answer_instruction = "Your response should be a comprehensive assessment or diagnosis based on the conversation history. Present it clearly and empathetically to the patient. Remind the patient that this is not a substitute for professional medical advice and they should consult a doctor."
            rationale_for_action = "Providing diagnosis as per user's choice after check-in."
        elif action_type == "ask_10th_question_check_in":
            answer_instruction = f"Your response should be: \"{TENTH_QUESTION_CHECK_IN_TEXT}\""
            rationale_for_action = "Offering user a choice to continue or get assessment (10th question check-in)."
        elif action_type == "consider_early_diagnosis":
            # LLM needs to decide if diagnosis is narrowed or if more questions are needed.
            answer_instruction = (
                "Based on your expert assessment of the conversation history: "
                "IF the differential diagnosis is now reasonably narrowed to 1-2 strong possibilities AND sufficient information has been gathered, "
                "THEN your response should be a comprehensive assessment or diagnosis, including next steps and a reminder to consult a doctor. "
                "ELSE (if more information is still critically needed or the diagnosis is still broad), your response should be the single most pertinent follow-up question."
            )
            # Rationale will be more complex for the LLM to fill in this case.
            rationale_for_action = "[LLM to state if providing early diagnosis due to narrowed scope, or if asking another question because more info is needed]"
        # Else, default (ask_follow_up) instructions are already set.
        
        question_or_permission = "Since the question count is not a multiple of 10, add a sentence to continue adding the question." if assistant_question_count % 10 != 0 else \
                                                "Since the question count is a multiple of 10, I should ask the member's permission to see if they want to provide more responses."
        
        prompt_for_llm = f"""You are an expert medical AI assistant.
Given the conversation history below (ending with the patient's latest message):

<conversation_history>
{formatted_history}
</conversation_history>

Your task is to generate a response that includes a 'think' section and an appropriate reply to the patient.

Follow these instructions precisely:
1.  Create a `<think>` section. Inside this section:
    a. A very brief summary (~50 words) of symptoms and other relevant medical facts presented so far. This will motivate the differential diagnosis below.
    b. Provide a concise differential diagnosis based on all information in the conversation history. List 2-3 potential conditions or areas to explore.
    c. State the "Question count for assistant (questions asked before this turn): {assistant_question_count}". {question_or_permission}
    d. Briefly state the rationale for your next action. This should align with the decision made (asking a question, offering check-in, providing diagnosis, or deciding between early diagnosis/more questions). Use: "{rationale_for_action}" 
    Replace the placeholder in [] with your reasoning.
2.  After the closing `</think>` tag, provide your response to the patient.
    {answer_instruction}

Your entire output should strictly follow this format:
<think>
Ok let us see. Here is a summary of the case so far: [Analysis of the symptoms and other relevant medical facts presented so far] 
Based on this, we have the differential diagnosis: [Your detailed differential diagnosis here]
Ok, lets think about what we should do next. [Your rationale here, replacing the placeholder if it's for LLM to fill]

Ok let me remember the Question count for assistant (questions asked before this turn): {assistant_question_count}. [Is this a multiple of 10?]][Your decision if we should continue questions or ask for permission to ask more questions.]
</think>
[Your response to the patient here]

Ensure your response to the patient is natural, empathetic, and directly addresses the instruction. Avoid any preamble before the `<think>` tag or after your response to the patient.
If the rationale placeholder is "[LLM to provide reasoning for the question it will ask]" or similar, replace it with your concise thought process for the question you formulate.
If the rationale is explicitly provided (e.g., "Providing diagnosis as per user's choice..."), use that.
"""
        return self._call_openai_llm_for_label(prompt_for_llm)

    def extract_examples_for_sft(self, source_conversations: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        extracted = []
        for conv_idx, conversation_history in enumerate(source_conversations):
            current_processing_history: List[Dict[str, str]] = []
            assistant_turns_count = 0 
            
            for turn_idx, turn_message in enumerate(conversation_history):
                current_processing_history.append(turn_message)

                if turn_message["role"] == "assistant":
                    assistant_turns_count += 1
                elif turn_message["role"] == "user":
                    sft_prompt_messages = list(current_processing_history) 
                    context_for_label_gen_inclusive = list(current_processing_history)

                    extracted.append({
                        "conversation_index": conv_idx, 
                        "turn_index": turn_idx, 
                        "history_for_sft_prompt": sft_prompt_messages,
                        "context_for_label_generation": context_for_label_gen_inclusive, 
                        "assistant_question_count": assistant_turns_count 
                    })
        logging.info(f"Extracted {len(extracted)} potential SFT examples from conversations.")
        return extracted

    def create_sft_dataset(self, 
                           all_source_conversations: List[List[Dict[str, str]]], 
                           target_min_examples: int, 
                           target_max_examples: int) -> Dataset:
        sft_dataset_prompts: List[List[Dict[str,str]]] = []
        sft_dataset_labels: List[str] = []

        if not all_source_conversations:
            logging.warning("No source conversations provided. Returning empty dataset.")
            return Dataset.from_dict({"prompt": [], "label": []})

        potential_examples = self.extract_examples_for_sft(all_source_conversations)
        random.shuffle(potential_examples)

        num_examples_to_generate = min(len(potential_examples), target_max_examples)
        if len(potential_examples) < target_min_examples:
             logging.warning(f"Number of potential examples ({len(potential_examples)}) is less than target_min_examples ({target_min_examples}). Will generate all potential examples.")
             num_examples_to_generate = len(potential_examples)
        elif num_examples_to_generate < target_min_examples : 
             logging.warning(f"target_max_examples ({target_max_examples}) is less than target_min_examples ({target_min_examples}). Will use target_min_examples if possible.")
             num_examples_to_generate = min(len(potential_examples), target_min_examples)

        logging.info(f"Attempting to generate ~{num_examples_to_generate} SFT examples.")

        for i in tqdm(range(num_examples_to_generate), desc="Generating SFT Dataset Labels"):
            example_data = potential_examples[i]
            sft_prompt = example_data["history_for_sft_prompt"]
            
            sft_label = self.generate_sft_label(
                example_data["context_for_label_generation"],
                example_data["assistant_question_count"]
            )
            
            sft_dataset_prompts.append(sft_prompt)
            sft_dataset_labels.append(sft_label)

            if (i + 1) % 50 == 0: 
                logging.info(f"Generated {i+1}/{num_examples_to_generate} labels...")

        if len(sft_dataset_prompts) < target_min_examples and len(potential_examples) >= target_min_examples:
            logging.warning(f"Generated {len(sft_dataset_prompts)} examples, less than target_min_examples ({target_min_examples}), though enough potential examples existed.")
        
        return Dataset.from_dict({"prompt": sft_dataset_prompts, "label": sft_dataset_labels})

    def save_dataset_to_disk(self, dataset: Dataset, output_path: str):
        try:
            dataset.save_to_disk(output_path)
            logging.info(f"Dataset successfully saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save dataset to {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate Supervised Fine-Tuning (SFT) training dataset for a medical AI assistant.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing conversations (columns: conversation_id, doctor_response, patient_response).")
    parser.add_argument("--output_path", type=str, required=True, help="Directory path to save the generated Hugging Face dataset.")
    parser.add_argument("--api_key", type=str, default=os.getenv(OPENAI_API_KEY_ENV_VAR), help=f"OpenAI API key. Defaults to environment variable {OPENAI_API_KEY_ENV_VAR}.")
    parser.add_argument("--model", type=str, default=LLM_MODEL_NAME_DEFAULT, help=f"Model to use for generating labels (e.g., models/gemini-2.0-flash, gpt-4). Default: {LLM_MODEL_NAME_DEFAULT}")
    parser.add_argument("--base_url", type=str, default=OPENAI_API_BASE_URL_DEFAULT, help=f"Base URL for the API. Default: {OPENAI_API_BASE_URL_DEFAULT}")
    parser.add_argument("--min_examples", type=int, default=TARGET_MIN_EXAMPLES_DEFAULT, help=f"Target minimum number of examples in the dataset. Default: {TARGET_MIN_EXAMPLES_DEFAULT}")
    parser.add_argument("--max_examples", type=int, default=TARGET_MAX_EXAMPLES_DEFAULT, help=f"Target maximum number of examples in the dataset. Default: {TARGET_MAX_EXAMPLES_DEFAULT}")
    parser.add_argument("--min_q_early_dx", type=int, default=MIN_QUESTIONS_FOR_CONSIDERING_EARLY_DIAGNOSIS, help=f"Min questions before LLM considers if early diagnosis is appropriate. Default: {MIN_QUESTIONS_FOR_CONSIDERING_EARLY_DIAGNOSIS}")
    
    args = parser.parse_args()

    # Update global config if overridden by CLI arg
    MIN_QUESTIONS_FOR_CONSIDERING_EARLY_DIAGNOSIS = args.min_q_early_dx

    if not args.api_key or args.api_key == "YOUR_OPENAI_API_KEY_HERE": 
        logging.error(f"OpenAI API key not provided or is a placeholder. Please set the {OPENAI_API_KEY_ENV_VAR} environment variable or use the --api_key argument.")
        return

    generator = SFTDatasetGenerator(api_key=args.api_key, model_name=args.model, base_url=args.base_url)

    logging.info(f"Loading conversations from: {args.csv_file}")
    source_conversations = generator.load_conversations_from_csv(args.csv_file)

    if not source_conversations:
        logging.error("No conversations loaded. Exiting.")
        return
    
    logging.info(f"Creating SFT dataset with target examples: {args.min_examples}-{args.max_examples}. Min questions for early Dx consideration: {MIN_QUESTIONS_FOR_CONSIDERING_EARLY_DIAGNOSIS}")
    fine_tuning_dataset = generator.create_sft_dataset(
        source_conversations,
        args.min_examples,
        args.max_examples
    )
    
    logging.info(f"\n--- Dataset Creation Complete ---")
    logging.info(f"Final number of examples in dataset: {len(fine_tuning_dataset)}")
    
    if len(fine_tuning_dataset) > 0:
        logging.info("\nSample of the first few examples:")
        for i in range(min(3, len(fine_tuning_dataset))):
            example = fine_tuning_dataset[i]
            prompt_summary = (f"History with {len(example['prompt'])} turns. "
                              f"Last turn by '{example['prompt'][-1]['role']}': "
                              f"{example['prompt'][-1]['content'][:60]}...")
            label_summary = f"{example['label'][:200].replace(chr(10), ' ')}..." 
            logging.info(f"\nExample {i+1}:")
            logging.info(f"  SFT Prompt Summary (model input): {prompt_summary}")
            logging.info(f"  SFT Label Preview (model target): {label_summary}")

        generator.save_dataset_to_disk(fine_tuning_dataset, args.output_path)
    else:
        logging.info("No data in the final SFT dataset. Nothing to save.")
    
    logging.info("Script finished.")

if __name__ == "__main__":
    main()
