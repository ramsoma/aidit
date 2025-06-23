import os
import json
import numpy as np
import pandas as pd
import logging

def get_conversation_length(cid, mean=20, sd=6):
    return np.random.normal(mean, sd)

def save_evaluation_results(conversations, metadata_dict, output_dir):
    """
    Saves all conversation examples and their metadata to files
    
    Args:
        conversations: Dictionary of conversations
        metadata_dict: Dictionary of conversation metadata
        output_dir: Directory to save results
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save conversations and metadata in JSON format
    with open(os.path.join(output_dir, "conversations.json"), "w") as f:
        json.dump(conversations, f, indent=2)
        
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_dict, f, indent=2)
    
    # Create summary CSV file
    summary_data = []
    for conv_id, meta in metadata_dict.items():
        row = {
            "conversation_id": conv_id,
            "type": meta["type"]
        }
        summary_data.append(row)
        
    # Convert to DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    
    print(f"Saved conversations and metadata to {output_dir}")


def create_training_chat_conversations(df, ids, cfg, system_prompt=None):
    """
    Creates multiple training examples from each conversation covering different scenarios:
    a) Assistant follows up with a question
    b) Assistant asks if member is okay to answer more questions
    c) Assistant provides a diagnosis
    
    Args:
        df: DataFrame containing conversation data
        ids: List of conversation IDs to process
        cfg: Configuration object with parameters
        
    Returns:
        tuple: (conversations_dict, metadata_dict) containing all generated training examples and metadata
    """
    all_conversations = {}
    all_metadata = {}  # Simplified metadata dictionary
    
    for cid in ids:
        group = df[df['conversation_id'] == cid]
        group = group.reset_index(drop=True)
        
        if len(group) == 0:
            logging.error(f"No conversation found for ID: {cid}")
            continue
            
        predicted_conv_length = get_conversation_length(cid, cfg.conversation_length_mean, cfg.conversation_length_sd)
        
        # Generate different examples from this conversation
        examples, examples_metadata = generate_conversation_examples(group, cid, predicted_conv_length, cfg, system_prompt=system_prompt)
        
        # Add examples to the result dictionaries
        for example_id, conversation in examples.items():
            all_conversations[example_id] = conversation
            all_metadata[example_id] = {
                "training_data_id": example_id,
                "conversation_id": cid,
                "type": examples_metadata[example_id]["type"]
            }
    
    return all_conversations, all_metadata


def generate_conversation_examples(group, cid, predicted_conv_length, cfg, system_prompt=None):
    """
    Generates multiple training examples from a single conversation.
    
    Args:
        group: DataFrame containing a single conversation
        cid: Conversation ID
        predicted_conv_length: Predicted length for this conversation
        cfg: Configuration object
        
    Returns:
        tuple: (examples dict, metadata dict) containing generated examples and their metadata
    """
    examples = {}
    metadata = {}
    
    # Example 1: Standard conversation with follow-up questions
    for i in range(1):
        examples[f"{cid}_standard_{i}"] = create_standard_conversation(group,  predicted_conv_length, system_prompt=system_prompt)
        metadata[f"{cid}_standard_{i}"] = {
            "type": "standard"
        }
    
    # Example 2: Early exit with diagnosis request
    for i in range(1):
        exit_percentage = np.random.uniform(0.3, 0.5)
        examples[f"{cid}_early_exit_{i}"] = create_early_exit_conversation(group, system_prompt=system_prompt)
        metadata[f"{cid}_early_exit_{i}"] = {
            "type": "early_exit"
        }
    
    # Example 3: Conversation with periodic check-ins
    for i in range(1):
        max_turns = np.random.randint(0, len(group))
        # round to nearest 10
        max_turns = int(round(max_turns / 10) * 10)
        examples[f"{cid}_check_ins_{i}"] = create_conversation_for_check_ins(group, turns=max_turns, system_prompt=system_prompt)
        metadata[f"{cid}_check_ins_{i}"] = {
            "type": "check_ins"
        }
    
    return examples, metadata


def create_standard_conversation(group, predicted_conv_length, system_prompt=None):
    """Creates a standard conversation example with natural follow-up questions"""
    conversation = []
    conversation.append({"role": "system", "content": system_prompt})
    
    # Limit to predicted length or actual length, whichever is smaller
    max_turns = min(len(group), predicted_conv_length)
    
    for index, row in group.iterrows():
        if index >= max_turns:
            break
            
        # Add doctor's response
        conversation.append({"role": "assistant", "content": row['doctor_response']})
        
        # Add patient's response
        conversation.append({"role": "user", "content": row['patient_response']})
        if index % 20 == 0:
            conversation.append({"role": "assistant", "content": "Are you able to answer a few more questions or would you like me to provide a diagnosis based on the current information?"})
            conversation.append({"role": "user", "content": "Yes, I can answer more questions."})

    return conversation


def create_early_exit_conversation(group, system_prompt=None):
    """Creates a conversation that ends early with a diagnosis request"""
    conversation = []
    conversation.append({"role": "system", "content": system_prompt})
    
    # Use approximately 30-90% of the conversation before exit
    exit_percentage = np.random.uniform(0.3, 0.5)
    early_exit_point = max(2, int(len(group) * exit_percentage))
    
    for index, row in group.iterrows():
        if index >= early_exit_point:
            break
            
        # Add doctor's response
        conversation.append({"role": "assistant", "content": row['doctor_response']})
        
        # Add patient's response
        conversation.append({"role": "user", "content": row['patient_response']})
    
    # Add request to end conversation
    conversation.append({"role": "assistant", "content": "Are you able to answer a few more questions or would you like me to provide a diagnosis based on the current information?"})
    conversation.append({"role": "user", "content": "I don't want to answer any more questions. Please provide a diagnosis."})
  
    return conversation


def create_conversation_for_check_ins(group, turns=10, system_prompt=None):
    """Creates a conversation with periodic check-ins to ask if user wants to continue
    
    Args:
        group: DataFrame containing conversation data
        cid: Conversation ID
        predicted_conv_length: Predicted length of the conversation
        cfg: Configuration object
        check_in_freq: Optional specific check-in frequency (if None, will check in at turns 10, 20, 30, etc.)
        
    Returns:
        conversation list
    """
    conversation = []
    conversation.append({"role": "system", "content": system_prompt})
    
    for index, row in group.iterrows():
        if index >= turns:
            break
            
        # Add doctor's response
        conversation.append({"role": "assistant", "content": row['doctor_response']})
        
        # Add patient's response
        conversation.append({"role": "user", "content": row['patient_response']})
                            
    return conversation
