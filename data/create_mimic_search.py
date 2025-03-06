"""
Retrieval Dataset Creator

This script creates training, development, and test datasets for the EHR retrieval task. 

The output format pairs each query with one positive and one negative document example.

Usage:
python data/create_mimic_search.py --configuration all_text --b 0.4 --k 0.9

    


"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import jsonlines
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_relevant_docs(filename: str) -> Dict[str, Set[str]]:
    """
    Extract relevant documents for each query from a TREC-format qrels file.
    
    Args:
        filename: Path to the qrels file
        
    Returns:
        Dictionary mapping query IDs to sets of relevant document IDs
    """
    gold_docs = {}
    try:
        with open(filename, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    logger.warning(f"Skipping malformed line in {filename}: {line.strip()}")
                    continue
                    
                qid, _, docid, rel = parts
                if int(rel) == 1:
                    gold_docs.setdefault(qid, set()).add(docid)
        
        logger.info(f"Loaded {len(gold_docs)} queries with relevant documents from {filename}")
        return gold_docs
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return {}
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return {}


def get_hard_negatives_docs(
    filename: str, 
    gold_docs: Dict[str, Set[str]], 
    max_negatives: int = 100
) -> Dict[str, Set[str]]:
    """
    Extract hard negative documents from BM25 results, excluding gold standard documents.
    
    Args:
        filename: Path to the BM25 results file
        gold_docs: Dictionary of gold standard relevant documents
        max_negatives: Maximum number of negative documents per query
        
    Returns:
        Dictionary mapping query IDs to sets of negative document IDs
    """
    neg_docs = {}
    try:
        with open(filename, 'r') as file:
            for line in file.readlines():
                elements = line.strip().split()
                if len(elements) < 3:
                    logger.warning(f"Skipping malformed line in {filename}: {line.strip()}")
                    continue
                
                # Parse TREC-format results (qid Q0 docid rank score runid)
                qid = elements[0]
                docid = elements[2]
                
                # Skip if we already have max negatives for this query
                if len(neg_docs.get(qid, set())) >= max_negatives:
                    continue
                
                # Skip if this is a relevant document
                if docid in gold_docs.get(qid, set()):
                    continue
                
                neg_docs.setdefault(qid, set()).add(docid)
        
        logger.info(f"Loaded negative documents for {len(neg_docs)} queries from {filename}")
        return neg_docs
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return {}
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return {}


def prepare_corpus(
    split: str,
    b: str,
    k: str,
    configuration: str,
    topics_dir: str = "topics",
    runs_dir: str = "runs",
    output_dir: str = "corpus_ehr"
) -> None:
    """
    Prepare a corpus of query-document pairs with positives and negatives.
    
    Args:
        split: Dataset split (train, dev, test, minitest)
        b: BM25 b parameter used in retrieval
        k: BM25 k1 parameter used in retrieval
        configuration: Configuration name for file paths
        topics_dir: Directory containing topic files
        runs_dir: Directory containing run files
        output_dir: Directory for output files
    """
    # Handle the 'minitest' special case
    qrel_gold = os.path.join(topics_dir, f"gold_{split}.txt")
    if split == 'minitest':
        qrel_gold = os.path.join(topics_dir, "gold_test.txt")
    
    qrel_bm25 = os.path.join(runs_dir, f"run_bm25_b_{b}_k1_{k}_top_1000_{configuration}_{split}.txt")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get relevant and negative documents
    gold_docs = get_relevant_docs(qrel_gold)
    neg_docs = get_hard_negatives_docs(qrel_bm25, gold_docs)
    
    # Create final dataset
    final_data = []
    for query in gold_docs:
        # if query not in neg_docs and split != 'minitest':
        #     logger.warning(f"No negative documents found for query {query}")
        #     continue
            
        try:
            entry = {
                "query_id": query, 
                "pos": list(gold_docs[query]), 
                "neg": {"bm25": list(neg_docs.get(query, set()))}
            }
            final_data.append(entry)
        except Exception as e:
            logger.error(f"Error processing query {query}: {e}")
    
    # Write output file
    output_file = os.path.join(output_dir, f"retrieval_{split}_{configuration}_posneg.jsonl")
    try:
        with jsonlines.open(output_file, 'w') as file:
            file.write_all(final_data)
        logger.info(f"Wrote {len(final_data)} entries to {output_file}")
    except Exception as e:
        logger.error(f"Error writing to {output_file}: {e}")


def load_posneg_dataset(configuration: str, corpus_dir: str = "corpus_ehr") -> Dict[str, List[Dict]]:
    """
    Load positive-negative document pairs for all splits.
    
    Args:
        configuration: Configuration name for file paths
        corpus_dir: Directory containing corpus files
        
    Returns:
        Dictionary mapping splits to lists of query-document pairs
    """
    logger.info("Loading positive and negative documents")
    all_corpus = {}
    
    for split in ['train', 'dev', 'test']:
        corpus_file = os.path.join(corpus_dir, f"retrieval_{split}_{configuration}_posneg.jsonl")
        try:
            with jsonlines.open(corpus_file) as reader:
                all_corpus[split] = list(reader)
            logger.info(f"Loaded {len(all_corpus[split])} entries from {corpus_file}")
        except FileNotFoundError:
            logger.error(f"File not found: {corpus_file}")
            all_corpus[split] = []
        except Exception as e:
            logger.error(f"Error reading {corpus_file}: {e}")
            all_corpus[split] = []
    
    return all_corpus


def load_corpus(configuration: str) -> Dict[str, str]:
    """
    Load document corpus mapping IDs to text content.
    
    Args:
        configuration: Configuration name for file paths
        
    Returns:
        Dictionary mapping document IDs to document text
    """
    logger.info("Loading document corpus")
    corpus = {}
    
    corpus_file = f"index/ehr_serialized_{configuration}/ehr_serialized_{configuration}.jsonl"
    try:
        with open(corpus_file, 'r') as file:
            for line in file:
                entry = json.loads(line)
                corpus[entry['id']] = entry['contents']
        logger.info(f"Loaded {len(corpus)} documents from {corpus_file}")
        return corpus
    except FileNotFoundError:
        logger.error(f"File not found: {corpus_file}")
        return {}
    except Exception as e:
        logger.error(f"Error reading {corpus_file}: {e}")
        return {}


def load_queries_dict(topics_dir: str = "topics") -> Dict[str, Dict[str, str]]:
    """
    Load query text for all splits.
    
    Args:
        topics_dir: Directory containing topic files
        
    Returns:
        Dictionary mapping splits to dictionaries of query ID to query text
    """
    logger.info("Loading queries")
    all_queries = {}
    
    for split in ['train', 'dev', 'test', 'minitest']:
        filename = os.path.join(topics_dir, f"query_id_{split}.tsv")
        try:
            tmp_queries = {}
            with open(filename, "r") as file:
                for line in file:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        logger.warning(f"Skipping malformed line in {filename}: {line.strip()}")
                        continue
                    qid, txt = parts
                    tmp_queries[qid] = txt
            all_queries[split] = tmp_queries
            logger.info(f"Loaded {len(tmp_queries)} queries from {filename}")
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            all_queries[split] = {}
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            all_queries[split] = {}
    
    return all_queries


def create_format_dataset(
    all_corpus: Dict[str, List[Dict]],
    corpus: Dict[str, str],
    all_queries: Dict[str, Dict[str, str]],
    output_dir: str = "corpus_ehr"
) -> None:
    """
    Create format dataset with 1 positive and 1 negative example per query.
    
    Args:
        all_corpus: Dictionary of query-document pairs by split
        corpus: Dictionary mapping document IDs to document text
        all_queries: Dictionary of query texts by split
        output_dir: Directory for output files
    """
    
    for split in [ 'test']:
    #for split in ['train', 'dev', 'test']:
        logger.info(f"Generating {split} set")
        entry_list = []
        
        for query_info in tqdm(all_corpus[split], desc=f"Processing {split} queries"):
            it_neg = 0
            neg_docs = query_info['neg']['bm25']  # List of negative documents
            
            if not neg_docs:
                logger.warning(f"No negative documents for query {query_info['query_id']}")
                continue
                
            # Iterate over positive documents
            max_pos = len(neg_docs)
            for pos in query_info['pos']:
                try:
                    # Get document texts
                    pos_txt = corpus.get(pos)
                    if not pos_txt:
                        logger.warning(f"Missing positive document {pos} in corpus")
                        continue
                        
                    neg_id = neg_docs[it_neg % max_pos]
                    neg_txt = corpus.get(neg_id)
                    if not neg_txt:
                        logger.warning(f"Missing negative document {neg_id} in corpus")
                        continue
                    
                    # Create entry with query, positive, and negative examples
                    entry = {
                        'qid': query_info['query_id'],
                        'q_txt': all_queries[split].get(query_info['query_id'], ""),
                        'posid': pos,
                        'pos_txt': pos_txt,
                        'negid': neg_id,
                        'neg_txt': neg_txt
                    }
                    entry_list.append(entry)
                    
                except Exception as e:
                    logger.error(f"Error processing entry {query_info['query_id']}: {e}")
                
                it_neg += 1
        
        # Write output file
        os.makedirs(output_dir, exist_ok=True)
        outfile = os.path.join(output_dir, f"treqs_posx1neg_{split}.jsonl")
        
        try:
            with jsonlines.open(outfile, 'w') as file:
                file.write_all(entry_list)
            logger.info(f"Wrote {len(entry_list)} entries to {outfile}")
        except Exception as e:
            logger.error(f"Error writing to {outfile}: {e}")


def main():
    """Main function to run the dataset creation pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create EHR retrieval datasets")
    parser.add_argument('--configuration', type=str, default='all_text',
                       help='Configuration identifier')
    parser.add_argument('--topics-dir', type=str, default='./data/mimic_search',
                       help='Directory containing topic files')
    parser.add_argument('--runs-dir', type=str, default='./runs',
                       help='Directory containing BM25 run files')
    parser.add_argument('--output-dir', type=str, default='./data/mimic_search/train',
                       help='Directory for output files')
    parser.add_argument('--b', type=str, default='0.2',
                       help='BM25 b parameter')
    parser.add_argument('--k', type=str, default='0.5',
                       help='BM25 k1 parameter')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Prepare corpus for all splits
    logger.info("Step 1: Preparing corpus with positive and negative examples")
    for split in ['test']:
    #for split in ['train', 'dev', 'test', 'minitest']:
        prepare_corpus(
            split=split,
            b=args.b,
            k=args.k,
            configuration=args.configuration,
            topics_dir=args.topics_dir,
            runs_dir=args.runs_dir,
            output_dir=args.output_dir
        )
    
    # Step 2: Create MSMARCO format dataset
    logger.info("Step 2: Creating format dataset")
    all_corpus = load_posneg_dataset(args.configuration, args.output_dir)
    corpus = load_corpus(args.configuration)
    all_queries = load_queries_dict(args.topics_dir)
    
    create_format_dataset(
        all_corpus=all_corpus,
        corpus=corpus,
        all_queries=all_queries,
        output_dir=args.output_dir
    )
    
    logger.info("Dataset creation completed successfully")


if __name__ == "__main__":
    main()