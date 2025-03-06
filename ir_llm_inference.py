"""
LLM Inference

Usage:
python ir_llm_inference.py --input data/mimic_search/all_test_html_non_guided.json --model meta-llama/Llama-2-7b-chat-hf --max-length 4096 --output_file metric_r1.pkl --metrics_file metrics.json
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import math
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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


def write_list(a_list: List[Any], file_name: str) -> None:
    """
    Write list to binary file using pickle.
    
    Args:
        a_list: List to write
        file_name: Path to the output file
    """
    import pickle
    with open(file_name, 'wb') as fp:
        pickle.dump(a_list, fp)
    print(f"Results saved to {file_name}")


def read_jsonl(filename: str) -> List[Dict[str, Any]]:
    """
    Read data from a JSONL file.
    
    Args:
        filename: Path to the JSONL file
        
    Returns:
        List of dictionaries parsed from the file
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


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
    patient_match = re.search(r'Passage:', text)
    question_match = re.search(r'Query:', text)
    
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

    
    # Get token IDs for "Yes" and "No"
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    # Run inference
    inference_list = []
    for i in tqdm(range(0, len(dataset), batch_size)):

        batch_prompts = dataset[i:i + batch_size]
    
        # Extract text and metadata
        batch_texts_original = [p["prompt"] for p in batch_prompts]
        batch_qids = [p.get("qid", "") for p in batch_prompts]
        batch_dids = [p.get("did", "") for p in batch_prompts]
        batch_expected = [p.get("completion", "") for p in batch_prompts]
        
        # Apply truncation to each text in the batch
        batch_texts = [truncate_text_parts(text, tokenizer, max_input_tokens) for text in batch_texts_original]
        
        # Tokenize the batch
        batch_encodings = tokenizer(
            batch_texts, 
            padding=True, 
            return_tensors="pt", 
            add_special_tokens=False
        )
        
        # Move batch to device
        batch_inputs = batch_encodings.input_ids.to(model.device)
        batch_attention_mask = batch_encodings.attention_mask.to(model.device)
        
        
        # Generate response
        with torch.no_grad():
            tokens = model.generate(
                batch_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=batch_attention_mask,
                temperature=temperature,
                return_dict_in_generate=True, 
                output_scores=True
            )
            
        # Process results for each item in batch
        generated_logits = tokens.scores
        first_gen_token_logits = generated_logits[0]  # Shape: [batch_size, vocab_size]

        for j in range(len(batch_prompts)):
            if j < len(first_gen_token_logits):  # Safety check
                # Extract logits for this item
                logit_yes = first_gen_token_logits[j][yes_id].cpu().numpy()
                logit_no = first_gen_token_logits[j][no_id].cpu().numpy()
                
                # Calculate probability score for "yes"
                # Calculate probability score for "yes" using the log-sum-exp trick for numerical stability
                max_logit = max(logit_yes, logit_no)
                exp_yes = math.exp(logit_yes - max_logit)
                exp_no = math.exp(logit_no - max_logit) 
                score_yes = (exp_yes / (exp_yes + exp_no)) * 100
                
                # Extract generated answer text for this item
                input_len = len(batch_encodings.input_ids[j]) - batch_encodings.attention_mask[j].sum().item()
                answer = tokenizer.decode(
                    tokens.sequences[j][input_len:], 
                    skip_special_tokens=True
                ).strip()
                
                # Add to inference list - store the original text, not the truncated one
                inference_list.append((
                    batch_qids[j], 
                    batch_dids[j], 
                    score_yes, 
                    batch_texts_original[j],
                    answer, 
                    batch_expected[j]
                ))


    # Return results
    return inference_list




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



def evaluate_results(results, metric_file, results_file, threshold=50.0):
    """
    Evaluate model performance from inference results.

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Evaluating results")
    
    
    # Prepare data for evaluation
    y_true = []
    y_pred = []
    y_score = []
    
    for qid, did, score_yes, input_text, answer, expected in results:
        # Convert to binary classification
        true_label = 1 if expected.lower() == "yes" else 0
        pred_label = 1 if score_yes >= threshold else 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_score.append(score_yes)
    
    # Calculate metrics
    metrics = {
        "total_examples": len(results),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "threshold_used": threshold
    }
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })
    
    # Calculate per-query metrics
    query_metrics = {}
    query_results = {}
    
    for qid, did, score_yes, input_text, answer, expected in results:
        if qid not in query_results:
            query_results[qid] = []
        
        true_label = 1 if expected.lower() == "yes" else 0
        pred_label = 1 if score_yes >= threshold else 0
        
        query_results[qid].append((true_label, pred_label, score_yes))
    
    for qid, items in query_results.items():
        q_true = [t for t, p, s in items]
        q_pred = [p for t, p, s in items]
        
        # Calculate accuracy for each query
        query_metrics[qid] = {
            "total": len(items),
            "accuracy": accuracy_score(q_true, q_pred)
        }
        
        # If the query has both positive and negative examples
        if sum(q_true) > 0 and sum(q_true) < len(q_true):
            query_metrics[qid].update({
                "precision": precision_score(q_true, q_pred),
                "recall": recall_score(q_true, q_pred),
                "f1": f1_score(q_true, q_pred)
            })
    
    metrics["per_query"] = query_metrics
    
    # Save metrics
    with open(metric_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    # Save results
    write_list(results, results_file) 

    return metrics


def main(
    input_file: str,
    output_file: str,
    metric_file: str,
    model_name: str = DEFAULT_MODEL,
    encoder_max_length: int = DEFAULT_MAX_LENGTH,
    temperature: float = DEFAULT_TEMPERATURE,
    use_4bit: bool = True,
    
) -> None:
    """
    Run the full inference and evaluation pipeline.

    """
    # Auto-generate output filename if not provided
    logger.info(f"Starting inference with model: {model_name}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output will be saved to: {output_file} and metrics to: {metric_file}")
    logger.info(f"Using max length: {encoder_max_length}.")
    
    # Verify max length is reasonable - ensure integer arithmetic
    effective_max_length = int(encoder_max_length-1) #int(encoder_max_length - max_new_tokens - 1)
    if effective_max_length <= 0:
        raise ValueError(
            f"Max length ({encoder_max_length}) minus max new tokens ({max_new_tokens}) "
        )
    logger.info(f"Effective max input length: {effective_max_length}")
    
    # Load data
    prompts = read_jsonl(input_file)
    
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
        prompts,
        encoder_max_length,
        temperature
    )

    # Evaluate results
    metrics = evaluate_results(
        results,
        metric_file=metric_file,
        results_file=output_file,
    )
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM inference and evaluation")
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input JSONL file with prompts')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save the output results')
    parser.add_argument('--metrics_file', type=str, required=True,
                        help='Path to save evaluation metrics')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Name or path of the model to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--use_4bit', action='store_true', 
                        help='Use 4-bit quantization')
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, 
                        help="Maximum sequence length for encoding")
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        output_file=args.output_file,
        metric_file=args.metrics_file,
        model_name=args.model,
        encoder_max_length=args.max_length,
        temperature=args.temperature,
        use_4bit=args.use_4bit,
    )