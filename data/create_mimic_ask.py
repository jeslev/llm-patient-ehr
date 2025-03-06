"""
Creation of MIMIC ASK from MIMIC III source

This script processes a MIMIC dataset containing patient questions and SQL queries.
It executes SQL queries against a SQLite database, formats the results,
and generates human-readable answers.

The dataset files are updated with a generated answer after querying the database.
"""

from datasets import DatasetDict, Dataset
import pandas as pd
import json
import sqlite3
from typing import Dict, List, Tuple, Any, Optional, Set
import os
from tqdm import tqdm


# Configuration
DB_PATH = "./mimicsql_db/mimic_qa.db"
DATASET_PATH = "./data/mimic_ask"
OUTPUT_DIR = "./data/mimic_ask"

# Dictionary to map column names to more readable formats
COLUMN_NAME_MAPPING = {
 'admittime': 'admission time',
 'dischtime': 'discharge time',
 'dob': 'birthday',
 'dod': 'date of death',
 'formulary drug cd': 'drug code',
 'itemid': 'item ID',
 'value unit': 'value',
}

# Special text replacements needed to handle complex formatting
TEXT_REPLACEMENTS = [
    ("MVR; ", "MVRPT "),
    ("GRAFT; ", "GRAFTPT  "),
    ("STERNOTOMY; ", "STERNOTOMYPT  "),
]

# Reverse replacements for final output
REVERSE_REPLACEMENTS = [
    ("MVRPT ", "MVR; "),
    ("GRAFTPT ", "GRAFT; "),
    ("STERNOTOMYPT ", "STERNOTOMY; "),
]


# Define mappings for specific patient admissions
SUBJECT_TO_ADM_MAPPING =  {2560: 126863, 6983: 173511, 81923: 101866, 29961: 196409, 3343: 193452, 62296: 193576, 15898: 163908, 7273: 151733,
 8990: 113207, 74032: 117458, 21796: 144540, 3623: 178444, 94762: 160707, 52012: 146937, 813: 159474, 18480: 156404, 23602: 152293,
 990: 184231, 16438: 152794, 9271: 181224, 2110: 135965, 42820: 137524, 55094: 135633, 29767: 135995, 74345: 154452, 10317: 156940,
 25167: 155314, 42067: 110395, 31066: 193361, 99936: 107913, 1121: 181293, 30011: 128610, 29541: 103957, 81254: 196317, 9575: 190574,
 93033: 103330, 17519: 154006, 65652: 195000, 71798: 122790, 24425: 119129, 17787: 184939, 4589: 101114, 91588: 173731, 7578: 135846,
 76446: 104538, 84129: 161253, 17570: 183034, 14755: 151669, 85673: 136522, 25543: 124588, 26285: 170242, 18351: 136878, 17595: 135289,
 18372: 148342, 43220: 117549, 15061: 191258, 74463: 106269, 22377: 120859, 1875: 122192, 8440: 137021, 11221: 100096, 53707: 195809,
 45962: 168649, 52118: 103211, 4342: 123907, 76508: 127570, 98220: 189416, 81085: 169845, 19420: 154375, 18112: 191779, 5027: 199776,
 19620: 130531, 8323: 152145, 18077: 142058, 19187: 154967, 32418: 193289, 64208: 184417, 42963: 177796, 22999: 157386, 28588: 141664,
 94756: 116857, 72353: 125843, 65759: 151565, 65982: 141119, 23733: 124364, 66411: 178264, 92796: 158687, 9258: 183354, 87275: 146248,
 73843: 158983, 14680: 146819, 3369: 126808, 17772: 122127, 26746: 179850, 12220: 180284, 83678: 172089, 4333: 155027, 3284: 119388,
 5506: 133761, 67965: 145657}


def load_json_to_dataset(
    json_dir: str,
    splits: List[str] = ['train', 'dev', 'test'],
    suffix: str = ''
) -> DatasetDict:
    """
    Load JSON files back into a Huggingface Dataset.
    
    Args:
        json_dir: Directory containing the JSON files
        splits: List of split names (corresponding to file names without .json extension)
        suffix: Suffix for JSON files
        
    Returns:
        DatasetDict containing the loaded datasets
    """
    datasets_dict = {}
    
    for split in splits:
        json_path = os.path.join(json_dir, f"{split}{suffix}.json")
        
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Skipping {split} split.")
            continue
        
        # Load JSON data
        print(f"Loading {split} data from {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create dataset
        dataset = Dataset.from_list(data)
        datasets_dict[split] = dataset
        print(f"Loaded {len(dataset)} records for {split} split")
    
    return DatasetDict(datasets_dict)

def ensure_output_directory(directory: str) -> None:
    """
    Ensure the output directory exists.
    
    Args:
        directory: Path to the output directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created output directory: {directory}")


def connect_to_database(db_path: str) -> sqlite3.Connection:
    """
    Connect to the SQLite database.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Connection to the database
    """
    try:
        connection = sqlite3.connect(db_path)
        return connection
    except sqlite3.Error as e:
        raise Exception(f"Error connecting to database: {e}")


def execute_sql_query(query: str, connection: sqlite3.Connection) -> Tuple[List[Any], Any]:
    """
    Execute a SQL query and return the results.
    
    Args:
        query: SQL query to execute
        connection: Database connection
        
    Returns:
        Tuple of (results, column descriptions)
    """
    cursor = connection.cursor()
    try:
        # Check if it's a SELECT from DEMOGRAPHIC
        is_demographic_query = (
            query.upper().strip().startswith("SELECT") and 
            "DEMOGRAPHIC" in query.upper()
        )
        
        results = cursor.execute(query).fetchall()
        
        # If this is a DEMOGRAPHIC query and we have results, we may need to filter
        if is_demographic_query and results and len(results) > 1:
            # We need to get the column index for SUBJECT_ID, HADM_ID, and ADMITTIME
            column_names = [desc[0] for desc in cursor.description]
            
            subj_idx = next((idx for idx, name in enumerate(column_names) if name.upper() == "SUBJECT_ID"), None)
            hadm_idx = next((idx for idx, name in enumerate(column_names) if name.upper() == "HADM_ID"), None)
            admit_idx = next((idx for idx, name in enumerate(column_names) if name.upper() == "ADMITTIME"), None)
            
            # If we have the necessary columns, apply filtering
            if subj_idx is not None and hadm_idx is not None:
                filtered_results = []
                
                for row in results:
                    subject_id = int(row[subj_idx]) if row[subj_idx] else None
                    hadm_id = int(row[hadm_idx]) if row[hadm_idx] else None
                    
                    if subject_id in SUBJECT_TO_ADM_MAPPING and hadm_id == SUBJECT_TO_ADM_MAPPING[subject_id]:
                        # This is the specific admission we want
                        filtered_results = [row]
                        break
                
                # If no specific admission matched, use the last one by ADMITTIME
                if not filtered_results and admit_idx is not None:
                    results = sorted(results, key=lambda x: x[admit_idx])
                    filtered_results = [results[-1]]  # Get the last one
                
                # If we successfully filtered, use the filtered results
                if filtered_results:
                    results = filtered_results
        
        return results, cursor.description
    except Exception as e:
        print(f"Error executing query: {query}")
        print(f"Error message: {e}")
        return [], None
    

def convert_query_results_to_dict(cursor_description: Any, results: List[Any]) -> Dict[str, List[str]]:
    """
    Convert SQL query results to a dictionary format.
    
    Args:
        cursor_description: Column descriptions from cursor
        results: Results from SQL query
        
    Returns:
        Dictionary mapping column names to lists of values
    """
    if not cursor_description or not results:
        return {}
    
    result_dict = {}
    for idx, col in enumerate(cursor_description):
        values = []
        for row in results:
            values.append(str(row[idx]))
        if values:
            result_dict[col[0]] = values
    
    return result_dict


def process_column_name(column_name: str) -> str:
    """
    Process a column name to make it more readable.
    
    Args:
        column_name: Original column name
        
    Returns:
        Processed column name
    """
    normalized_name = column_name.strip().lower().replace("_", " ")
    return COLUMN_NAME_MAPPING.get(normalized_name, normalized_name)


def apply_text_replacements(text: str, replacements: List[Tuple[str, str]]) -> str:
    """
    Apply a list of text replacements to a string.
    
    Args:
        text: Text to modify
        replacements: List of (pattern, replacement) tuples
        
    Returns:
        Modified text
    """
    for pattern, replacement in replacements:
        text = text.replace(pattern, replacement)
    return text


def date_to_text(date_time: str) -> str:
    """
    Convert date-time string to readable text format.
    
    Args:
        date_time: Date-time string to format
        
    Returns:
        Formatted date-time string
    """
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    try:
        # Handle different date formats
        if ' ' in date_time:
            # Format: YYYY-MM-DD HH:MM:SS
            (date, time) = date_time.split(" ")
            (year, month, day) = date.split("-")
            (hour, minutes, seconds) = time.split(":")
            return f"{months[int(month) - 1]} {day.lstrip('0')} {year} at {hour}h{minutes}"
        elif '-' in date_time:
            # Format: YYYY-MM-DD
            (year, month, day) = date_time.split("-")
            return f"{months[int(month) - 1]} {day.lstrip('0')} {year}"
        else:
            return date_time
    except Exception as e:
        # If parsing fails, return original
        print(f"Date parsing error for '{date_time}': {e}")
        return date_time


def format_answer_value(column: str, value: str) -> str:
    """
    Format a specific answer value based on its column.
    
    Args:
        column: Column name
        value: Value to format
        
    Returns:
        Formatted value
    """
    # Handle gender formatting
    if column.strip() == 'gender':
        if value.strip() == 'M':
            return "masculine"
        if value.strip() == 'F':
            return "feminine"
    
    # Handle date formatting for specific columns
    date_columns = ['admission time', 'discharge time', 'birthday', 'date of death']
    if column.strip() in date_columns:
        return date_to_text(value.strip())
    
    return value.strip()


def process_answers(answers: List[Dict[str, Any]]) -> List[str]:
    """
    Process SQL query answers into readable text.
    
    Args:
        answers: List of dictionaries containing SQL query results
        
    Returns:
        List of formatted answer strings
    """
    formatted_answers = []
    
    for answer_dict in answers:
        if not answer_dict or not answer_dict.get('answer'):
            formatted_answers.append("")
            continue
            
        # Group all values by column across all answer entries
        grouped_answers = {}
        
        for answer_entry in answer_dict.get('answer', []):
            # Apply text replacements first
            processed_entry = apply_text_replacements(answer_entry, TEXT_REPLACEMENTS)
            
            # Split by semicolon to get separate column:value pairs
            parts = processed_entry.split("; ")
            
            for part in parts:
                column_parts = part.split(":")
                if len(column_parts) >= 2:
                    column = process_column_name(column_parts[0])
                    value = ":".join(column_parts[1:])  # Handle values that might contain colons
                    
                    # Apply reverse replacements
                    for pattern, replacement in REVERSE_REPLACEMENTS:
                        value = value.replace(pattern, replacement)
                    
                    # Format specific columns
                    formatted_value = format_answer_value(column, value)
                    
                    # Add to the grouped answers
                    values = grouped_answers.get(column, set())
                    values.add(formatted_value)
                    grouped_answers[column] = values
        
        # Format into text
        answer_text_parts = []
        for column, values in grouped_answers.items():
            formatted_values = ", ".join(list(values))
            answer_text_parts.append(f"{column}: {formatted_values}")
        
        formatted_answer = "; ".join(answer_text_parts)
        formatted_answers.append(formatted_answer)
    
    return formatted_answers


def execute_and_process_sql_queries(
    dataset_split: Dataset, 
    db_connection: sqlite3.Connection
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Execute SQL queries in a dataset split and process the results.
    
    Args:
        dataset_split: Dataset split containing SQL queries
        db_connection: Database connection
        
    Returns:
        Tuple of (raw query results, formatted answers)
    """
    raw_query_results = []
    
    print(f"Executing SQL queries for {len(dataset_split)} records...")
    for query in tqdm(dataset_split['sql_query']):
        results, description = execute_sql_query(query, db_connection)
        
        if description:
            result_dict = convert_query_results_to_dict(description, results)
            if result_dict:
                raw_query_results.append(result_dict)
            else:
                raw_query_results.append({})
        else:
            raw_query_results.append({})
    
    # Clean and format the answers
    print("Processing query results into readable answers...")
    
    # Prepare data for processing - each item should have the format {'answer': [list of formatted entries]}
    data_for_processing = []
    for result in raw_query_results:
        if not result:
            data_for_processing.append({'answer': []})
            continue
            
        # Each result dictionary is one SQL result set with multiple columns
        # Format each row as "COL1:value1; COL2:value2; ..." 
        formatted_rows = []
        
        # Determine number of rows from the first column's values
        if result:
            first_col = next(iter(result.values()))
            num_rows = len(first_col)
            
            # For each row in the results
            for row_idx in range(num_rows):
                row_parts = []
                # Get values for each column in this row
                for col, values in result.items():
                    if row_idx < len(values):
                        row_parts.append(f"{col}:{values[row_idx]}")
                
                formatted_rows.append("; ".join(row_parts))
            
        data_for_processing.append({'answer': formatted_rows})
    
    formatted_answers = process_answers(data_for_processing)
    
    return raw_query_results, formatted_answers



def compare_answers(original_answers: List[str], new_answers: List[str]) -> None:
    """
    Compare original answers with newly generated answers.
    
    Args:
        original_answers: List of original answers
        new_answers: List of new answers
    """
    if len(original_answers) != len(new_answers):
        print(f"Warning: Answer count mismatch. Original: {len(original_answers)}, New: {len(new_answers)}")
    
    exact_matches = 0
    different_answers = []
    
    for i, (orig, new) in enumerate(zip(original_answers, new_answers)):
        if orig == new:
            exact_matches += 1
        else:
            different_answers.append((i, orig, new))
    
    match_percentage = (exact_matches / len(original_answers)) * 100 if original_answers else 0
    
    print(f"Answer comparison results:")
    print(f"  - Total answers: {len(original_answers)}")
    print(f"  - Exact matches: {exact_matches} ({match_percentage:.2f}%)")
    print(f"  - Different answers: {len(different_answers)}")
    
    # Print a few examples of differences
    if different_answers:
        print("\nExample differences (first 5):")
        for i, (index, orig, new) in enumerate(different_answers[:5]):
            print(f"\nExample {i+1} (index {index}):")
            print(f"  Original: {orig}")
            print(f"  New     : {new}")


def save_dataset_to_json(dataset: pd.DataFrame, output_path: str) -> None:
    """
    Save a dataset to a JSON file.
    
    Args:
        dataset: Dataset to save
        output_path: Path to save the dataset to
    """
    print(f"Saving dataset to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset.to_dict(orient='records'), f, indent=2)



def process_dataset() -> None:
    """Main function to process the MIMIC dataset."""
    # Ensure output directory exists
    ensure_output_directory(OUTPUT_DIR)
    
    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}")
    dataset = load_json_to_dataset(DATASET_PATH, suffix='_orig')
    
    # Connect to database
    db_connection = connect_to_database(DB_PATH)
    
    # Process each split
    processed_data = {}
    
    for split_name in ['test', 'dev', 'train']:
        print(f"\nProcessing {split_name} split...")
        split_data = dataset[split_name]
        
        # Extract needed columns
        required_columns = ['question', 'sql_query', 'question_treqs_complexity', 'patient_id']
        
        df = pd.DataFrame({col: split_data[col] for col in required_columns})
        
        # Execute SQL queries and process results
        raw_results, new_answers = execute_and_process_sql_queries(split_data, db_connection)
        
        # Add new answers to dataframe
        df['answer'] = new_answers       
        
        # Save processed dataset
        processed_data[split_name] = df
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        save_dataset_to_json(df, output_path)
    
    db_connection.close()
    print("\nAll processing complete!")


if __name__ == "__main__":
    process_dataset()


