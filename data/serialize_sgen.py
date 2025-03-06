"""

Script to generate the sgen method serialization from the corpus
"""

import os
import json
import pickle
from typing import Dict, List, Any, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM



# Model settings
MODEL_PATH = 'meta-llama/Llama-2-7b-chat-hf'
MAX_NEW_TOKENS = 1500
TEMPERATURE = 0.0  # 0.0 for deterministic generation

DATASET_FILE = 'data/mimic_ask/all_train_html_guided.json'
PREFIX_PROMPT = "Identify critical values and ranges of the table to solve the task.\n\n"

# HTML cleanup replacements
HTML_REPLACEMENTS = [
    ('</table>', ''),
    ('<table border="1">', ''),
    ('<table>', ''),
    ('ANSWER:', '')
]


def load_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the LLM model and tokenizer.
    
    Args:
        model_path: Path to the pre-trained model
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        print(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map='auto',
            torch_dtype=torch.float16  
        )
        
        return model, tokenizer
    
    except Exception as e:
        raise RuntimeError(f"Error loading model and tokenizer: {e}")


def load_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """
    Load and parse the dataset from a JSONL file.
    
    Args:
        dataset_file: Path to the dataset file
        
    Returns:
        List of dictionaries containing the dataset examples
    """
    try:
        dataset = []
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    dataset.append(json.loads(line))
        
        print(f"Loaded {len(dataset)} examples from {dataset_file}")
        return dataset
    
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")


def preprocess_input_text(prompt: str, input_text: str, replacements: List[Tuple[str, str]]) -> str:
    """
    Preprocess the input text by removing HTML tags and adding the prompt.
    
    Args:
        prompt: Prefix prompt to add
        input_text: The input text to preprocess
        replacements: List of (old, new) text replacements to apply
        
    Returns:
        Preprocessed input text
    """
    # Combine prompt with input text
    full_text = prompt + input_text
    
    # Apply all replacements
    for old, new in replacements:
        full_text = full_text.replace(old, new)
    
    return full_text


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    Generate an answer using the model.
    
    Args:
        model: The LLM model
        tokenizer: The tokenizer
        input_text: Preprocessed input text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated answer as string
    """
    # Tokenize the input
    tokens = tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
    inputs, att_mask = tokens.input_ids, tokens.attention_mask
    
    # Move tensors to the appropriate device
    inputs = inputs.to(model.device)
    att_mask = att_mask.to(model.device)
    
    # Generate
    with torch.no_grad():
        output_tokens = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=att_mask,
        )
    
    # Extract only the newly generated tokens
    input_len = inputs.size()[-1]
    answer = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True).strip()
    
    return answer


def save_results(results: Dict[int, str], output_file: str) -> None:
    """
    Save the generated answers to a file.
    
    Args:
        results: Dictionary mapping example index to generated answer
        output_file: Path to save the results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"Saving {len(results)} results to {output_file}")
        with open(output_file, 'wb') as fp:
            pickle.dump(results, fp)
    
    except Exception as e:
        print(f"Error saving results: {e}")


def main() -> None:
    """Main function to run the inference."""
    OUTPUT_FILE = DATASET_FILE.replace("html", "sgen")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
        
        # Load dataset
        dataset = load_dataset(DATASET_FILE)
        
        # Generate answers
        all_answers = {}
        
        for idx, example in enumerate(tqdm(dataset,)):
            try:
                # Preprocess input
                input_text = preprocess_input_text(PREFIX_PROMPT, example['prompt'], HTML_REPLACEMENTS)
                
                # Generate answer
                answer = generate_answer(
                    model,
                    tokenizer,
                    input_text,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                )
                
                # Store the answer
                all_answers[idx] = answer
            
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
        
        # Save results
        save_results(all_answers, OUTPUT_FILE)
        
        print("Inference completed successfully!")
    
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()