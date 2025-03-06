"""
LLM Inference

Usage:
python qa_llm_inference.py --input data/mimic_ask/all_train_text_guided.json --model meta-llama/Llama-2-7b-chat-hf --max-length 4096

"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple

import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
from evaluate import load
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
ACCESS_TOKEN = None # HuggingFace access token
DEFAULT_MODEL = 'meta-llama/Llama-2-7b-chat-hf'
DEFAULT_MAX_LENGTH = 4096
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_BATCH_SIZE = 1



def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the examples
    """
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        if not data:
            raise ValueError(f"No data found in {file_path}")
            
        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def setup_model_and_tokenizer(
    model_name: str, 
    encoder_max_length: int,
    use_4bit: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and configure the model and tokenizer.
    
    Args:
        model_name: Name or path of the model to load
        encoder_max_length: Maximum sequence length for the tokenizer
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        logger.info(f"Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            model_max_length=encoder_max_length, 
            token=ACCESS_TOKEN
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading model: {model_name}")
        
        # Configure quantization if requested
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_config,
            token=ACCESS_TOKEN
        )
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error setting up model and tokenizer: {e}")
        raise


def truncate_text_parts(text: str, tokenizer: AutoTokenizer, max_input_tokens: int) -> str:
    """
    Truncate the text by splitting it into parts and prioritizing the beginning and end.
    
    Args:
        text: Input text to truncate
        tokenizer: Tokenizer to use for counting tokens
        max_input_tokens: Maximum number of tokens allowed
        
    Returns:
        Truncated text
    """
    # Split the text into parts using regex
    patient_match = re.search(r'PATIENT:', text)
    question_match = re.search(r'QUESTION:', text)
    
    # Default split (if markers not found)
    beginning = text
    middle = ""
    end = ""
    
    # If we found both markers, split accordingly
    if patient_match and question_match:
        beginning = text[:patient_match.end()]
        middle = text[patient_match.end():question_match.start()]
        end = text[question_match.start():]
    # If only found patient marker, split there
    elif patient_match:
        beginning = text[:patient_match.end()]
        middle = text[patient_match.end():]
    
    # Tokenize each part to get token counts
    beginning_tokens = tokenizer.encode(beginning, add_special_tokens=False)
    middle_tokens = tokenizer.encode(middle, add_special_tokens=False)
    end_tokens = tokenizer.encode(end, add_special_tokens=False)
    
    # Check if we need to truncate
    total_tokens = len(beginning_tokens) + len(middle_tokens) + len(end_tokens)
    
    if total_tokens <= max_input_tokens:
        return text  # No truncation needed
    
    # Strategy: Keep the full beginning and end, truncate the middle
    tokens_for_middle = max_input_tokens - len(beginning_tokens) - len(end_tokens)
    
    # If we can't keep any middle tokens, we need to truncate the end
    if tokens_for_middle <= 0:
        # Try to keep at least half of the beginning
        beginning_to_keep = min(len(beginning_tokens), max_input_tokens // 2)
        end_to_keep = max_input_tokens - beginning_to_keep
        
        # Ensure we don't try to keep more end tokens than we have
        end_to_keep = min(end_to_keep, len(end_tokens))
        
        # Get truncated beginning and end parts
        truncated_beginning = tokenizer.decode(beginning_tokens[:beginning_to_keep])
        truncated_end = tokenizer.decode(end_tokens[:end_to_keep])
        
        return truncated_beginning + truncated_end
    
    # Truncate the middle part
    truncated_middle = tokenizer.decode(middle_tokens[:tokens_for_middle])
    
    # Combine parts
    return beginning + truncated_middle + end


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict[str, Any]],
    max_length: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> List[Dict[str, Any]]:
    """
    Run inference on the dataset with smart text truncation.
    
    Args:
        model: The LLM model
        tokenizer: The tokenizer
        dataset: List of examples with 'prompt' and 'completion' fields
        max_length: Maximum sequence length for encoding
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        batch_size: Batch size for inference
        
    Returns:
        List of examples with added 'generated_answer' field
    """
    try:
        # Calculate maximum input length
        # Convert to int to ensure we're working with integer values
        max_input_tokens = int(max_length - 1)#int(max_length - max_new_tokens - 1)
        
        # Validate the max input tokens
        if max_input_tokens <= 0:
            raise ValueError(
                f"Max input tokens ({max_input_tokens}) must be positive. "
                #f"Reduce max_new_tokens or increase max_length."
            )
        
        logger.info(f"Using max input tokens: {max_input_tokens}")
        
        # Create DataLoader with the raw dataset
        dataset_obj = Dataset.from_dict({
            'prompt': [item['prompt'] for item in dataset],
            'completion': [item['completion'] for item in dataset]
        })
        
        loader = DataLoader(
            dataset_obj, 
            shuffle=False, 
            num_workers=0, 
            batch_size=batch_size
        )
        
        # Run inference
        results = []
        for i, batch in enumerate(tqdm(loader, desc="Running inference")):
            item = dataset[i]  # Original item with all metadata
            
            prompt = batch['prompt'][0]  # Get prompt text
            golden_answer = batch['completion'][0]  # Get golden answer
            
            # Truncate the text if needed using our custom function
            truncated_prompt = truncate_text_parts(prompt, tokenizer, max_input_tokens)
            
            # Get token counts for reporting
            original_tokens = len(tokenizer.encode(prompt))
            truncated_tokens = len(tokenizer.encode(truncated_prompt))
            was_truncated = original_tokens != truncated_tokens
            
            # Tokenize with careful limits
            encoding = tokenizer(
                truncated_prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=max_input_tokens,  # Explicitly limit to our calculated max
                truncation=True  # Just in case our function didn't truncate enough
            )
            
            inputs = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            
            # Double-check the input size is within limits
            if inputs.size(1) > max_input_tokens:
                logger.warning(f"Input still too long after truncation: {inputs.size(1)} tokens. Forcing hard truncation.")
                inputs = inputs[:, :max_input_tokens]
                attention_mask = attention_mask[:, :max_input_tokens]
            
            # Generate response
            with torch.no_grad():
                output_tokens = model.generate(
                    inputs.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0.0),
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode only the new tokens
            input_len = inputs.size(1)
            answer = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True)
            
            # Add to results
            item_with_prediction = item.copy()
            item_with_prediction['generated_answer'] = answer
            item_with_prediction['token_counts'] = {
                "original": original_tokens,
                "truncated": truncated_tokens,
                "was_truncated": was_truncated
            }
            
            results.append(item_with_prediction)
            
            # Log first example and periodic samples
            if i == 0 or i % 50 == 0:
                logger.info(f"Example {i}:")
                logger.info(f"  Original tokens: {original_tokens}")
                logger.info(f"  Truncated tokens: {truncated_tokens}")
                logger.info(f"  Was truncated: {was_truncated}")
                logger.info(f"  Generated answer: {answer[:50]}...")
        
            
        return results
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

def compute_rouge_scores(
    predictions: List[str], 
    references: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores for the predictions.
    
    Args:
        predictions: List of generated answers
        references: List of reference (golden) answers
        
    Returns:
        Dictionary of ROUGE metrics
    """
    try:
        # Load ROUGE metric
        rouge = load("rouge")
        
        # Compute scores
        results = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        
        # Round results for readability
        rounded_results = {k: round(v * 100, 2) for k, v in results.items()}
        
        return rounded_results
    
    except Exception as e:
        logger.error(f"Error computing ROUGE scores: {e}")
        raise


def save_results(
    results: List[Dict[str, Any]], 
    output_file: str
) -> None:
    """
    Save results to a JSONL file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to save the results
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


def create_output_filename(input_file: str, model_name: str) -> str:
    """
    Create an output filename based on the input file and model name.
    
    Args:
        input_file: Path to the input file
        model_name: Name of the model used
        
    Returns:
        Output filename
    """
    # Extract the base name of the input file without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Extract the model short name (last part of the path)
    model_short_name = model_name.split('/')[-1]
    
    # Create the output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Create the output filename
    return f"output/results_{base_name}_{model_short_name}.jsonl"


def main(
    input_file: str,
    model_name: str = DEFAULT_MODEL,
    encoder_max_length: int = DEFAULT_MAX_LENGTH,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    use_4bit: bool = True,
    output_file: Optional[str] = None
) -> None:
    """
    Run the full inference and evaluation pipeline.
    
    Args:
        input_file: Path to the input JSONL file
        model_name: Name or path of the model to use
        encoder_max_length: Maximum sequence length for encoding
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        use_4bit: Whether to use 4-bit quantization
        output_file: Path to save the results (or None to auto-generate)
    """
    try:
        # Auto-generate output filename if not provided
        if output_file is None:
            output_file = create_output_filename(input_file, model_name)
        
        logger.info(f"Starting inference with model: {model_name}")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output will be saved to: {output_file}")
        logger.info(f"Using max length: {encoder_max_length}, max new tokens: {max_new_tokens}")
        
        # Verify max length is reasonable - ensure integer arithmetic
        effective_max_length = int(encoder_max_length-1) #int(encoder_max_length - max_new_tokens - 1)
        if effective_max_length <= 0:
            raise ValueError(
                f"Max length ({encoder_max_length}) minus max new tokens ({max_new_tokens}) "
            )
        logger.info(f"Effective max input length: {effective_max_length}")
        
        # Load data
        data = load_jsonl(input_file)
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(
            model_name,
            encoder_max_length,
            use_4bit
        )
        
        # Run inference with smart text truncation
        results = run_inference(
            model,
            tokenizer,
            data,
            encoder_max_length,
            max_new_tokens,
            temperature
        )
        
        # Extract predictions and references for ROUGE calculation
        predictions = [item['generated_answer'] for item in results]
        references = [item['completion'] for item in results]
        
        # Compute ROUGE scores
        rouge_scores = compute_rouge_scores(predictions, references)
        
        # Log ROUGE scores
        logger.info("ROUGE Scores:")
        for metric, score in rouge_scores.items():
            logger.info(f"  {metric}: {score}")
        
        # Add ROUGE scores to each result
        for item in results:
            item['rouge_scores'] = rouge_scores
        
        # Save results
        save_results(results, output_file)
        
        logger.info("Inference completed successfully")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    
    finally:
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM inference and evaluation")
    
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--output", default=None, help="Output file path (optional)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, 
                        help="Maximum sequence length for encoding")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, 
                        help="Sampling temperature")
    parser.add_argument("--no-4bit", action="store_true", 
                        help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        model_name=args.model,
        encoder_max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_4bit=not args.no_4bit,
        output_file=args.output
    )