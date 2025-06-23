import re
import json
import numpy as np
from collections import Counter
from typing import Optional, List, Dict
from datasets import load_from_disk, Dataset
# Assume these are initialized and available in your environment
# from sentence_transformers import SentenceTransformer, util

# --- Constants ---
THINK_TAG_OPEN = "<think>"
THINK_TAG_CLOSE = "</think>"
ANSWER_TAG_OPEN = "<answer>"
ANSWER_TAG_CLOSE = "</answer>"

# --- Helper Functions to Robustly Extract Content ---
def calculate_reward(c, prompts, utterance):
  reward = 0
  # Encourage questions early in the conversation
  if c == "question" and len(prompts) < 5:
    reward += 5 # Higher reward for early questions
  elif c == "question" and len(prompts) < 10:
    reward += 2 # Higher reward for early questions
  # Penalize never-ending question chains
  if c == "question":
    reward += 1 # Penalty for excessive questions
  # Penalize multiple questions in a single utterance
  if c == "multiple question":
    num_questions = utterance.count("?")
    reward -= (2 + (num_questions - 1))

  # Penalize out-of-topic responses
  if c == "other":
    reward -= 5

  # Encourage Clinical Diagnosis/Explanation
  clinical_labels = ["clinical diagnosis", "clinical explanation"]
  if c in clinical_labels and len(prompts) > 10:
    reward += 3

  # Penalize Clinical Diagnosis/Explanation without data collection

  if c in clinical_labels and len(prompts) < 5:
    reward -= 5

  # Adjust reward for utterance length
  utterance_length = len(utterance.split())

  if utterance_length > 20: # Discourage overly long utterances
    reward -= 1

  elif utterance_length < 3: # Discourage overly short utterances
    reward -= 1

  return reward

def classify(response, labels):
  try:
    results = classifier(response, candidate_labels=labels,

    batch_size=None)

    class_ = results['labels'][np.argmax(results['scores'])]
    return class_
  except:
    return "other"

def extract_content(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """
    Extracts content between a start and end tag.
    Returns None if tags are not found or content is empty.
    """
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    start_idx += len(start_tag)

    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        return None

    content = text[start_idx:end_idx].strip()
    return content if content else None

def extract_thinking_content(text: str) -> Optional[str]:
    """Extracts content from within <think>...</think> tags."""
    return extract_content(text, THINK_TAG_OPEN, THINK_TAG_CLOSE)

def extract_answer_content(text: str) -> Optional[str]:
    """All content outside of <think>...</think> tags is considered answer content."""
    thinking_content = extract_thinking_content(text)
    if thinking_content is None:
        return text
    else:
        return text.replace(thinking_content, "").replace("<think>", "").replace("</think>", "").strip()

def load_rubric_from_str(rubric_item_str):
  rubric = {} # Or some default rubric
  try:
      # Clean up potential markdown code blocks around JSON
      cleaned_rubric_str = rubric_item_str.replace("```json\n", "").replace("\n```", "").replace("```", "")
      rubric = json.loads(cleaned_rubric_str)
  except json.JSONDecodeError:
      rubric['think'] = None
      # The rubric string should match the answer.
      rubric['answer'] = rubric_item_str
  return rubric

def reward_for_question(answer_content_for_classification, labels, prompt_hist):
  if answer_content_for_classification is None or not answer_content_for_classification.strip():
            # If no answer content, or it's empty, classify as "other" or assign a specific penalty
            classification = "other"
            utterance_for_calc_reward = "" # Or some default string
  else:
      utterance_for_calc_reward = answer_content_for_classification
      # Ensure your `classify` function can handle potentially empty strings if not caught above
      classification = classify(utterance_for_calc_reward, labels)
      print('----------')

  # Calculate reward based on classification, prompt history, and utterance
  # Ensure your `calculate_reward` can handle `prompt_hist` (list of dicts)
  reward_val = calculate_reward(classification, prompt_hist, utterance_for_calc_reward)
  print(answer_content_for_classification, classification, reward_val)
  return reward_val / 5.0

def cosine_sim(a, b, sent_transformer):
  return util.cos_sim(sent_transformer.encode(a), sent_transformer.encode(b)).item()

def answer_reward_func(prompts: List[List[Dict[str,str]]],
                                completions: List[List[Dict[str, str]]],
                                rubrics: List[Dict],**kwargs) -> List[float]:
    """
    Reward function that checks if the completion (answer part) has a specific format/intent.
    Uses your `classify` and `calculate_reward` helpers.
    """
    labels = ["question", "multiple questions", "clinical diagnosis", "clinical explanation", "other"]
    metadata = kwargs.get("metadata", {})
    rewards_out = []
    for prompt_hist, comp_hist, rubric_item_str, imd in zip(prompts, completions, rubrics, metadata):
        rubric = load_rubric_from_str(rubric_item_str)
        full_completion_text = comp_hist[-1]['content']
        # Extract only the <answer> part for classification
        answer_content_for_classification = extract_answer_content(full_completion_text)

        if imd['type'] == "standard":
          reward_val = reward_for_question(answer_content_for_classification, labels, prompt_hist)
          reward_val += max([cosine_sim(answer_content_for_classification, ans, sent_transformer)
                           for ans in rubric['answer']]) if type(rubric['answer']) is list \
                           else cosine_sim(answer_content_for_classification, rubric['answer'], sent_transformer)
        elif imd['type']=="checkin":
          classfication = classify(answer_content_for_classification, ['request to ask more questions', 'other'])
          reward_val = 1 if classfication == 'request to ask more questions' else 0
        else:
          # providing a rubric
          reward_val = max([cosine_sim(answer_content_for_classification, ans, sent_transformer)
                           for ans in rubric['answer']]) if type(rubric['answer']) is list \
                           else cosine_sim(answer_content_for_classification, rubric['answer'], sent_transformer)

        rewards_out.append(reward_val)

    return (np.array(rewards_out))


def thinking_reward_func(prompts: List[List[Dict[str,str]]],
                                     completions: List[List[Dict[str, str]]],
                                     rubrics: List[Dict], # Assuming rubrics are passed and match batch order
                                     **kwargs) -> List[float]:
    """
    Rewards based on semantic similarity of thinking and answer parts to a rubric.
    Assumes `sent_transformer` and `util.cos_sim` are available.
    """
    rewards = []
    # Placeholder for sentence transformer - replace with your actual model
    # from sentence_transformers import SentenceTransformer, util
    # sent_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    for comp_hist, rubric_item_str in zip(completions, rubrics):
        response_text = comp_hist[-1]['content']

        # Robustly parse rubric
        rubric = load_rubric_from_str(rubric_item_str)

        thinking_content = extract_thinking_content(response_text)

        total_reward = 0.0
        num_components = 0

        if thinking_content and len(thinking_content.strip()) > 0 and 'think' in rubric:
            # Ensure rubric['differential_diagnosis'] is a string or list of strings
            rubric_diag_str = str(rubric['think'])
            diff_diagnosis_embedding = sent_transformer.encode(rubric_diag_str)
            response_thinking_embedding = sent_transformer.encode(thinking_content)
            diagnosis_reward = util.cos_sim(diff_diagnosis_embedding, response_thinking_embedding).item()
            total_reward += diagnosis_reward
            num_components +=1
        elif (thinking_content is None or len(thinking_content.strip()) == 0) and 'think' in rubric: # Penalize missing thinking if expected
            total_reward -= 1

        rewards.append(total_reward)

    return np.array(rewards)


