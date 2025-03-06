"""
Data preparation script for LLM input.

This script processes MIMIC data into instruction-based prompt for LLMs.
It extracts patient information using different serialization methods and creates guided or
non-guided instruction formats for questions and answers.

The script supports various serialization methods:
- text: Plain text representation of patient data
- html: HTML-formatted patient data
- sgen: Structured generation format of patient data
"""

import os
import json
from typing import Dict, List, Literal, Optional, Union, Any

import pandas as pd
from datasets import load_dataset, Dataset


# Constants
ROOT_PATH = './data/mimic_ask'
SPLITS = ['train', 'dev', 'test']
INSTRUCTION_FORMATS = ['guided', 'non_guided']
SERIALIZATION_METHODS = ['text', 'html'] #, 'sgen']
SERIALIZED_CORPUS = 'mimicsql_db/list_serialized_corpus.json'
PREFIX_CORPUS = 'all'


def read_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSON dataset file.

    Args:
        file_path: Path to the JSON file containing patient data

    Returns:
        List of dictionaries with patient data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading JSON dataset from {file_path}: {e}")
        return []


def create_instruction_format(
    answer: str,
    system_message: str
) -> Dict[str, str]:
    """
    Convert a data sample to the instruction format.

    Args:
        answer: The answer text
        system_message: The formatted system message with patient info and question

    Returns:
        Dictionary with 'prompt' and 'completion' keys
    """
    return {
        "prompt": f"{system_message}",
        "completion": f"{answer}"
    }


def process_file(
    data_serialized: Dict[str, Dict[str, str]], 
    key_dict: str, 
    patient_list: List[Dict[str, Any]], 
    system_message: str
) -> List[Dict[str, str]]:
    """
    Process patient data and create instruction format datasets.

    Args:
        data_serialized: Dictionary of serialized patient data
        key_dict: The key to access the correct serialization format
        patient_list: List of patient records with questions and answers
        system_message: Template for the system message

    Returns:
        List of processed examples in instruction format
    """
    processed_dataset = []
    
    for patient in patient_list:
        # Convert patient_id to string and ensure it's an integer first to handle format variations
        patient_id = str(int(patient['patient_id']))
        
        # Extract relevant information
        question = patient['question']
        answer = patient['answer']
        
        # Get the background information using the serialization method
        if patient_id in data_serialized and key_dict in data_serialized[patient_id]:
            background = data_serialized[patient_id][key_dict]
            
            # Format the system message with background and question
            formatted_message = system_message.format(background, question)
            
            # Create and add the instruction format example
            processed_dataset.append(create_instruction_format(answer, formatted_message))
        else:
            print(f"Warning: Patient ID {patient_id} not found in serialized data or missing {key_dict} key")
    
    return processed_dataset


def save_dataset(dataset: List[Dict[str, str]], output_file: str) -> None:
    """
    Save the dataset to a JSON lines file.

    Args:
        dataset: The dataset to save
        output_file: Path to the output JSON file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save as JSON Lines format (each line is a valid JSON object)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        print(f"Dataset successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving dataset to {output_file}: {e}")


def get_system_prompts() -> Dict[str, str]:
    """
    Define system prompts for different instruction formats.
    
    Returns:
        Dictionary mapping instruction type to system prompt
    """
    return {
        'non_guided': """You are an extremely helpful healthcare assistant, referring to the given passage above, provide the correct answer.
        
        PATIENT: {}

        QUESTION: {}

        ANSWER: """,
        
        'guided': """You are a highly skilled healthcare assistant. Follow these steps to accurately extract the information needed to answer the question: 
        1. Carefully read the question to understand what specific information is being asked.
        2. Review the patient profile and identify the relevant features that address the question.
        3. Extract the information corresponding to the relevant features from the patient profile.
        4. Format your answer as 'feature name': value1, value2, ensuring that you include all required values.
        Always ensure that your response is concise, precise, and based only on the provided patient profile.
        
        PATIENT: {}

        QUESTION: {}

        ANSWER: """
    }


def main() -> None:
    """Main function to process all datasets."""
    
    # Get system prompts for different instruction formats
    system_messages = get_system_prompts()
    
    # Define the mapping between serialization method and dictionary key
    serialization_key_mapping = {
        'text': 'template',
        'html': 'html_short_g2',
        'sgen': 'sgen',
    }
    
    # Load the serialized corpus containing patient data
    try:
        with open(SERIALIZED_CORPUS, 'r', encoding='utf-8') as json_file:
            data_serialized = json.load(json_file)
        print(f"Successfully loaded serialized corpus from {SERIALIZED_CORPUS}")
    except Exception as e:
        print(f"Error loading serialized corpus: {e}")
        return
    
    # Process each combination of parameters
    for serialization_method in SERIALIZATION_METHODS:
        key_dict = serialization_key_mapping[serialization_method]
        
        for split in SPLITS:
            # Load patient data for the current split
            split_file_path = os.path.join(ROOT_PATH, f'{split}.json')
            patient_list = read_json_dataset(split_file_path)
            
            if not patient_list:
                print(f"Skipping empty or invalid patient list for {split}")
                continue
                
            for instruction_format in INSTRUCTION_FORMATS:
                # Get the appropriate system message for this instruction format
                system_message = system_messages[instruction_format]
                
                # Process the patient data
                dataset = process_file(data_serialized, key_dict, patient_list, system_message)
                
                if dataset:
                    # Print a sample for debugging
                    # if len(dataset) > 34:
                    #     print(f"Sample from {split}, {instruction_format}, {serialization_method}:")
                    #     print(dataset[34])
                    
                    # Define output file path
                    output_file = os.path.join(ROOT_PATH, f'{PREFIX_CORPUS}_{split}_{serialization_method}_{instruction_format}.json')
                    
                    # Save the processed dataset
                    save_dataset(dataset, output_file)
                else:
                    print(f"No data processed for {split}, {instruction_format}, {serialization_method}")


if __name__ == "__main__":
    main()