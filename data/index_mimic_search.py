"""
Pyserini BM25 Search Pipeline

This script performs the following steps:
1. Converts JSON data to JSONL format for Pyserini
2. Indexes the data using Pyserini
3. Performs BM25 search using queries from a file
4. Evaluates the search results using trec_eval

Usage:
    python data/index_mimic_search.py --input ./mimicsql_db/list_serialized_corpus.json --output-dir runs --feature all --topk 10
"""


import json
import os
import time
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm


os.environ["PATH"] += os.pathsep + "/PATH/TO/YOUR/JDK/bin"
os.environ["JAVA_OPTS"] = "-Xms512M -Xmx4G"



def create_directory(directory_path: str) -> None:
    """Create a directory if it doesn't exist, clearing it if it does."""
    path = Path(directory_path)
    
    # Remove directory if it exists
    if path.exists():
        shutil.rmtree(path)
    
    # Create the directory
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {directory_path}")


def convert_json_to_jsonl(
    input_file: str, 
    output_file: str, 
    field_name: str,
) -> int:
    """
    Convert JSON data to JSONL format for Pyserini.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSONL file
        field_name: Field name to extract from the JSON
        
    Returns:
        Number of records processed
    """
    print(f"Converting {input_file} to JSONL format...")
    corpus = []

    # Read corpus and extract the specified field
    try:
        with open(input_file, 'r') as file:
            data = json.load(file)
            
            # Process data based on its structure
            if isinstance(data, dict):
                # Handle dictionary format (key-value pairs)
                for entry_id, entry in data.items():
                    
                    if field_name in entry:
                        corpus.append({
                            'id': entry['h_adm_id'], 
                            'contents': entry[field_name]
                        })
            elif isinstance(data, list):
                # Handle list format (array of objects)
                for entry in data:
                    if 'id' in entry and field_name in entry:
                        corpus.append({
                            'id': entry['id'], 
                            'contents': entry[field_name]
                        })
            else:
                print(f"Error: Unexpected JSON structure in {input_file}.")
                return 0
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return 0
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}.")
        return 0
    except KeyError:
        print(f"Error: Required fields not found in the input file.")
        return 0

    # Write new corpus in JSONL format
    try:
        with open(output_file, "w") as file:
            for entry in tqdm(corpus, desc="Writing JSONL"):
                json.dump(entry, file)
                file.write("\n")
    except IOError:
        print(f"Error: Could not write to {output_file}.")
        return 0

    print(f"Converted {len(corpus)} records to {output_file}")
    return len(corpus)


def run_pyserini_indexer(
    input_dir: str,
    index_path: str,
    threads: int = 12
) -> bool:
    """
    Run Pyserini indexer on the JSONL file.
    
    Args:
        input_dir: Directory containing the JSONL file
        index_path: Directory to store the index
        threads: Number of threads to use for indexing
        
    Returns:
        True if indexing was successful, False otherwise
    """
    print(f"Indexing documents from {input_dir} to {index_path}...")
    
    try:
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", input_dir,
            "--index", index_path,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(threads),
            #"--storePositions", "--storeDocvectors", "--storeRaw"
        ]
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        print(f"Indexing completed successfully to {index_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during indexing: {e}")
        print(f"Standard error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error during indexing: {e}")
        return False


def load_queries(query_file: str) -> Tuple[List[str], List[str]]:
    """
    Load queries from a TSV file.
    
    Args:
        query_file: Path to the query file (TSV format: qid<tab>query)
        
    Returns:
        Tuple of (queries, query IDs)
    """
    queries = []
    query_ids = []
    
    try:
        with open(query_file, 'r') as file:
            for line in file.readlines():
                qid, query = line.strip().split("\t")
                queries.append(query)
                query_ids.append(qid)
        
        print(f"Loaded {len(queries)} queries from {query_file}")
        return queries, query_ids
    
    except FileNotFoundError:
        print(f"Error: Query file {query_file} not found.")
        return [], []
    except Exception as e:
        print(f"Error loading queries: {e}")
        return [], []


def search_bm25(
    searcher: Any,
    queries: List[str],
    query_ids: List[str],
    output_file: str,
    topk: int = 100,
    b: float = 0.75,
    k1: float = 1.2
) -> bool:
    """
    Perform BM25 search with the given parameters.
    
    Args:
        searcher: Pyserini searcher object
        queries: List of query strings
        query_ids: List of query IDs
        output_file: Path to output file (TREC format)
        topk: Number of top documents to retrieve
        b: BM25 'b' parameter
        k1: BM25 'k1' parameter
        
    Returns:
        True if search was successful, False otherwise
    """
    print(f"Performing BM25 search with b={b}, k1={k1}, topk={topk}...")
    
    try:
        # Set BM25 parameters
        searcher.set_bm25(k1, b)
        
        # Open output file for writing
        with open(output_file, 'w') as file:
            for (query, qid) in tqdm(zip(queries, query_ids), total=len(queries), desc="Searching"):
                results = searcher.search(query, topk)
                
                for i, result in enumerate(results):
                    line = f'{qid} Q0 {result.docid} {i+1:3} {result.score:.5f} inutero\n'
                    file.write(line)
        
        print(f"Search results written to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error during search: {e}")
        return False


def evaluate_results(
    qrel_file: str,
    run_file: str,
    output_file: str
) -> bool:
    """
    Evaluate search results using trec_eval.
    
    Args:
        qrel_file: Path to qrel file (ground truth)
        run_file: Path to run file (search results)
        output_file: Path to output file for evaluation metrics
        
    Returns:
        True if evaluation was successful, False otherwise
    """
    print(f"Evaluating results using trec_eval...")
    
    try:
        cmd = [
            "python", "-m", "pyserini.eval.trec_eval",
            "-m", "map", "-m", "P", "-m", "recall",
            "-m", "ndcg_cut.5,10,20,100",
            qrel_file, run_file
        ]
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Write evaluation results to output file
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        
        print(f"Evaluation metrics written to {output_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        print(f"Standard error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
        return False


def run_pipeline(
    input_file: str,
    output_dir: str,
    feature: str,
    query_file: str,
    qrel_file: str,
    methods: List[str] = ['template', 'html_short_g2'],
    topk: int = 1000,
    b: float = 0.4,
    k1: float = 0.9,
    threads: int = 12,
    split: str = 'test',
) -> None:
    """
    Run the complete pipeline from JSON conversion to evaluation.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory for output files
        feature: Feature to use (all or avg)
        query_file: Path to query file
        qrel_file: Path to qrel file
        methods: List of methods to extract from JSON
        topk: Number of top documents to retrieve
        b: BM25 'b' parameter
        k1: BM25 'k1' parameter
        threads: Number of threads for indexing
        split: Split to use (train, dev or test)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create index directory
    index_base_dir = "index"
    os.makedirs(index_base_dir, exist_ok=True)
    
    # Load queries
    queries, query_ids = load_queries(query_file)
    if not queries:
        print("No queries loaded. Exiting.")
        return
    
    method_map = {
        'template': 'text',
        'html_short_g2': 'html',
    }

    # Process each method
    for _method in methods:
        method = method_map[_method]
        print(f"\n{'='*50}")
        print(f"Processing method: {method}")
        print(f"{'='*50}\n")
        
        # Define paths
        index_dir = f"{index_base_dir}/ehr_serialized_{feature}_{method}"
        jsonl_dir = f"{index_dir}"
        jsonl_file = f"{jsonl_dir}/ehr_serialized_{feature}_{method}.jsonl"
        
        # Create directories
        create_directory(jsonl_dir)
        
        # Convert JSON to JSONL
        records = convert_json_to_jsonl(input_file, jsonl_file, _method)
        if records == 0:
            print(f"No records processed for method {method}. Skipping.")
            continue
        
        # Index documents
        if not run_pyserini_indexer(jsonl_dir, index_dir, threads):
            print(f"Indexing failed for method {method}. Skipping.")
            continue
        
        # Import Pyserini searcher here to avoid errors if imports fail
        try:
            from pyserini.search.lucene import LuceneSearcher
            searcher = LuceneSearcher(index_dir)
        except ImportError:
            print("Error: Could not import Pyserini. Make sure it is installed.")
            continue
        except Exception as e:
            print(f"Error initializing searcher for {index_dir}: {e}")
            continue
        
        # Define output files
        run_file = f"{output_dir}/run_bm25_b_{b}_k1_{k1}_top_{topk}_{feature}_{method}_{split}.txt"
        metric_file = f"{output_dir}/metric_run_bm25_b_{b}_k1_{k1}_top_{topk}_{feature}_{method}_{split}.output"
        
        # Perform search
        if not search_bm25(searcher, queries, query_ids, run_file, topk, b, k1):
            print(f"Search failed for method {method}. Skipping evaluation.")
            continue
        
        # Evaluate results
        if not evaluate_results(qrel_file, run_file, metric_file):
            print(f"Evaluation failed for method {method}.")
            continue
        
        print(f"\nCompleted processing for method {method}")
    
    print("\nPipeline completed successfully!")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pyserini BM25 Search Pipeline")
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to the input JSON file')
    parser.add_argument('--output-dir', type=str, default='runs',
                       help='Directory for output files')
    parser.add_argument('--feature', type=str, default='all',
                       help='Which features to use: all or avg')
    parser.add_argument('--split', type=str, default='test',
                       help='Split to use: test, dev or train')
    parser.add_argument('--query-file', type=str, default='./data/mimic_search/query_id_test.tsv',
                       help='Path to the query file')
    parser.add_argument('--qrel-file', type=str, default='./data/mimic_search/gold_test.txt',
                       help='Path to the qrel file')
    parser.add_argument('--methods', type=str, nargs='+', default=['template', 'html_short_g2'],
                       help='Methods to extract from JSON')
    parser.add_argument('--topk', type=int, default=1000,
                       help='Number of top documents to retrieve')
    parser.add_argument('--b', type=float, default=0.2,
                       help='BM25 b parameter')
    parser.add_argument('--k1', type=float, default=1.7,
                       help='BM25 k1 parameter')
    parser.add_argument('--threads', type=int, default=12,
                       help='Number of threads for indexing')
    
    return parser.parse_args()


if __name__ == "__main__":
    
    # Parse arguments
    args = parse_arguments()
    
    # Run the pipeline
    run_pipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        feature=args.feature,
        query_file=args.query_file,
        qrel_file=args.qrel_file,
        methods=args.methods,
        topk=args.topk,
        b=args.b,
        k1=args.k1,
        threads=args.threads,
        split=args.split,
    )
    

