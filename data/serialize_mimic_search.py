"""
EHR Prompt Generator for LLM Input.

This script processes EHR data into instruction-based prompts for LLMs.
It extracts patient information and creates prompts that ask whether a given passage answers a query.
The script outputs JSONL files with prompts and their expected completions (yes/no).
"""


import json
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm


# Constants
DEFAULT_MAX_RERANK = 200
ROOT_PATH = './data/mimic_search'
SPLITS = ['test'] #[ 'dev', 'test', 'minitest']
INSTRUCTION_FORMATS = ['guided', 'non_guided']
SERIALIZATION_METHODS = ['text', 'html'] #, 'sgen']
FEATURE = 'all'

SERIALIZED_CORPUS = 'mimicsql_db/list_serialized_corpus.json'




def write_jsonl(filename: str, content: List[Dict[str, Any]]) -> None:
    """
    Write list of dictionaries to JSONL file.
    
    Args:
        filename: Path to the output file
        content: List of dictionaries to write
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        for entry in content:
            json.dump(entry, file)
            file.write('\n')
    print(f"Successfully wrote {len(content)} examples to {filename}")


def read_jsonl(filename: str) -> List[Dict[str, Any]]:
    """
    Read JSONL file and return list of dictionaries.
    
    Args:
        filename: Path to the JSONL file
        
    Returns:
        List of dictionaries parsed from the file
    """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def write_list(a_list: List[Any], file_name: str) -> None:
    """
    Write list to binary file using pickle.
    
    Args:
        a_list: List to write
        file_name: Path to the output file
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as fp:
        pickle.dump(a_list, fp)
        print(f'Done writing list to {file_name}')


def read_list(file_name: str) -> List[Any]:
    """
    Read list from binary file using pickle.
    
    Args:
        file_name: Path to the binary file
        
    Returns:
        List read from the file
    """
    with open(file_name, 'rb') as fp:
        return pickle.load(fp)


def load_corpus(corpus_path: str) -> Dict[str, str]:
    """
    Load document corpus from JSONL file.
    
    Args:
        corpus_path: Path to the corpus JSONL file
        
    Returns:
        Dictionary mapping document IDs to document contents
    """
    print(f"Loading document corpus from {corpus_path}")
    corpus = {}
    with open(corpus_path, 'r') as file:
        for line in file.readlines():
            entry = json.loads(line)
            corpus[entry['id']] = entry['contents']
    return corpus


def load_queries_dict(queries_path: str = None, 
                     ) -> Dict[str, str]:
    """
    Load queries from TSV files.
    
    Args:
        queries_path: Path to the queries TSV file         
    Returns:
        Dictionary mapping  query IDs and texts
    """
    print("Loading queries")
    all_queries = {}
    if queries_path is None:
        return None
        
    with open(queries_path, "r") as file:
        for line in file.readlines():
            qid, txt = line.strip().split("\t")
            all_queries[qid] = txt
    return all_queries


def load_run(filename: str) -> Dict[str, List[str]]:
    """
    Load BM25 run file that contains query-document pairs.
    
    Args:
        filename: Path to the BM25 run file
        
    Returns:
        Dictionary mapping query IDs to lists of document IDs
    """
    run = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            elems = line.strip().split(" ")
            b_qid, b_docid = False, False
            for el in elems:
                if len(el) == 0 or el == "Q0":
                    continue
                if not b_qid:
                    b_qid = True
                    qid = el.strip()
                    continue
                if not b_docid:
                    b_docid = True
                    docid = el.strip()
            
            # Add to the dictionary
            current_docs = run.get(qid, [])
            current_docs.append(docid)
            run[qid] = current_docs
    return run


def read_qrel(filename: str) -> Dict[str, List[str]]:
    """
    Read relevance judgements file.
    
    Args:
        filename: Path to the relevance judgements file
        
    Returns:
        Dictionary mapping query IDs to lists of relevant document IDs
    """
    qrels = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            elems = line.split("\t")
            qid, did = elems[0], elems[2]
            rel_docs = qrels.get(qid, [])
            rel_docs.append(did)
            qrels[qid] = rel_docs
    return qrels


def load_examples_test_pointwise(
    queries: Dict[str, str], 
    corpus: Dict[str, str], 
    docs: Optional[Dict[str, List[str]]] = None, 
    filtered_queries: Optional[List[int]] = None, 
    max_rerank: int = DEFAULT_MAX_RERANK
) -> Tuple[List[Tuple[str, str, Tuple[str, str]]], List[Tuple[str, str]]]:
    """
    Create examples of query-document pairs for testing.
    
    Args:
        queries: Dictionary mapping query IDs to query texts
        corpus: Dictionary mapping document IDs to document texts
        docs: Dictionary mapping query IDs to lists of document IDs (optional)
        filtered_queries: Range of query indices to include (optional)
        max_rerank: Maximum number of documents to rerank per query
        
    Returns:
        Tuple containing:
            - List of (query_text, doc_text, (query_id, doc_id)) tuples
            - List of (query_id, doc_id) tuples
    """
    input_examples = []
    query_passage_ids = []
    
    list_qids = sorted(queries.keys())
    for (cnt_id, q_id) in enumerate(list_qids):
        q_txt = queries[q_id]
        
        # Filter queries if specified
        if filtered_queries is not None:
            if (int(cnt_id) < int(filtered_queries[0]) or int(cnt_id) > int(filtered_queries[1])):
                continue
        
        # Process documents
        if docs is None:  # Full retrieval
            all_keys = sorted(corpus.keys())
            for d_id in all_keys:
                d_txt = corpus[d_id]
                input_examples.append((q_txt, d_txt, (q_id, d_id)))
                query_passage_ids.append((q_id, d_id))
        else:  # Rerank
            doc_cnt = 0
            for d_id in docs[q_id]:
                if doc_cnt >= max_rerank:
                    break
                input_examples.append((q_txt, corpus[d_id], (q_id, d_id)))
                query_passage_ids.append((q_id, d_id))
                doc_cnt += 1
                
    return input_examples, query_passage_ids


def load_corpus_and_dataset(
    corpus_filename: str,
    queries_path: Optional[str] = None, 
    run_filename: Optional[str] = None
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], Optional[Dict[str, Dict[str, List[str]]]]]:
    """
    Load corpus and queries from files.
    
    Args:
        corpus_filename: Path to corpus
        queries_path: Path to training queries file (optional)
        run_filename: Format string for BM25 run files (optional)
        
    Returns:
        Tuple containing:
            - Dictionary mapping document IDs to document texts
            - Dictionary mapping split names to dictionaries of query IDs and texts
            - Dictionary mapping split names to dictionaries of query IDs and document ID lists (optional)
    """
    corpus = load_corpus(corpus_filename)
    # Load queries
    all_queries = load_queries_dict(queries_path)
    
    # Recover doc ids from first-passage ranking (BM25)
    corpus_run = None
    if run_filename is not None:
        corpus_run = load_run(run_filename)
    
    print(f"Total docs in corpus: {len(corpus)}. Total queries: {len(all_queries)}")
    return corpus, all_queries, corpus_run


def get_system_prompts() -> Dict[str, str]:
    """
    Define system prompts for different instruction formats.
    
    Returns:
        Dictionary mapping instruction type to system prompt
    """
    return {
        'non_guided': """You are an extremely helpful healthcare assistant, analyze the following passage and the query.
        
        Passage: {}

        Query: {}

        Answer 'Yes' or 'No': """,
        
        'guided': """You are a healthcare assistant responsible to decide if a patient profile matches the query. Follow this process:
1.Understand the query by identifying the key information or criteria it is asking for.
 2. Examine the patient profile. The profile is in html format followed by a summary. Search for any features (e.g.,  demographics, diagnosis) that align with the query.
 3. Determine whether the patient’s information matches or is relevant to the query.

 Only consider the facts in the profile when making your decision.
        
        Passage: {}

        Query: {}

        Answer 'Yes' or 'No': """,
    }



# Prompt Generation
def generate_prompt(query, doc, instruction):

    system_messages = get_system_prompts()
    
    # Combine everything into one prompt
    prompt = system_messages[instruction].format(doc, query)
    return prompt



# Main Function
def generate_prompts(
    output_file: str,
    corpus_path: str,
    qrels_path: str,
    queries_path: str,
    run_filename: Optional[str] = None,
    filtered_queries: Optional[List[int]] = None,
    max_rerank: int = DEFAULT_MAX_RERANK,
    instruction: str = False, 
) -> None:
    """
    Generate prompts for LLM and save them to a JSONL file.
    
    Args:
        output_file: Path to save the output JSONL file
        corpus_path: Path to corpus file with all documents
        qrels_path: Path to qrels file for getting relevance judgments
        queries_path: Path to test queries file
        run_filename: Path to BM25 run file, optional for reranking
        filtered_queries: Range of query IDs to filter, optional
        max_rerank: Maximum number of documents to rerank per query
        guided: If it is guided instruction or not
    """
    print(f"Generating prompts and writing results to {output_file}")
    
    # Load corpus and queries
    corpus, all_queries, docs_for_reranking = load_corpus_and_dataset(
        corpus_path, queries_path=queries_path, run_filename=run_filename
    )
    
    # Load relevance judgments to determine correct answers
    qrels = read_qrel(qrels_path)
    
    results = []
    
    # Process each split ('test' in this case)
    input_examples, query_passage_ids = load_examples_test_pointwise(
        all_queries, corpus, docs=docs_for_reranking,
        filtered_queries=filtered_queries, max_rerank=max_rerank
    )
    
    print(f"Total examples for {split}: {len(input_examples)}")
    
    # Generate prompts for each example
    for i, (query, doc, ids) in enumerate(tqdm(input_examples)):
        qid, did = ids
        
        # Generate the prompt
        prompt = generate_prompt(query, doc, instruction)
        
        # Determine the correct answer (yes or no)
        is_relevant = did in qrels.get(qid, [])
        completion = "yes" if is_relevant else "no"
        
        # Add to results
        results.append({
            "prompt": prompt,
            "completion": completion,
            "qid": qid,
            "did": did
        })
    
    # Write results to JSONL file
    write_jsonl(output_file, results)


if __name__ == "__main__":
    
    # Given a feature, generate all configurations
    #  This is, serialization method, guided and non guided, and all splits 
    # # Example usage

    b = "0.2"
    k1 = "1.7"
    topk = "10"

    # Optional filtering of queries (e.g., for parallel processing)
    filtered_queries = [63, 126]  # Process queries between index 63 and 126

    for serialization in SERIALIZATION_METHODS:
        for split in SPLITS:
            test_queries = f"data/mimic_search/query_id_{split}.tsv"
            results_bm25run = f"runs/run_bm25_b_{b}_k1_{k1}_top_{topk}_{FEATURE}_{serialization}_{split}.txt"
            corpus_path = f'index/ehr_serialized_{FEATURE}_{serialization}/ehr_serialized_{FEATURE}_{serialization}.jsonl'
            test_qrels = os.path.join(ROOT_PATH, 'gold_test.txt')
            for instruction in INSTRUCTION_FORMATS:
                output_path = os.path.join(ROOT_PATH, f'{FEATURE}_{split}_{serialization}_{instruction}.json')
                # Generate prompts
                generate_prompts(
                    output_file=output_path,
                    corpus_path=corpus_path,
                    qrels_path=test_qrels,
                    queries_path=test_queries,
                    run_filename=results_bm25run,
                    filtered_queries=filtered_queries,
                    max_rerank=DEFAULT_MAX_RERANK,
                    instruction=instruction,
                )
