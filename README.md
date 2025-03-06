# LLMs on Tabular Electronic Health Records: Data Extraction and Retrieval 🏥

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/Conference-Accepted-success)](https://link-to-your-paper)

Official repository for our paper **"Evaluating LLM Abilities to Understand Tabular Electronic Health Records: A Comprehensive Study of Patient Data Extraction and Retrieval"** accepted at ECIR 2025. This repository contains code and instructions for creating datasets and running models for two main tasks:

1. **MIMIC ASK**: A patient data extraction task on Electronic Health Records (EHR)
2. **MIMIC SEARCH**: An information retrieval task on EHR data


## 📝 Citation

If you use this code or the datasets in your research, please cite our paper:

```bibtex
@inproceedings{lovon2025evaluating,
  title={Evaluating LLM Abilities to Understand Tabular Electronic Health Records: A Comprehensive Study of Patient Data Extraction and Retrieval},
  author={Lovon, Jesus and Mouysset, Martin and Oleiwan, Jo and Moreno, Jose G and Damase-Michel, Christine and Tamine, Lynda},
  booktitle={European Conference on Information Retrieval},
  year={2025}
}
```


## 📋 Table of Contents

- [Dataset Creation](#dataset-creation)
  - [MIMICSQL Dataset Generation](#mimicsql-dataset-generation)
  - [MIMIC ASK (Data Extraction Task)](#mimic-ask-qa-task)
  - [MIMIC SEARCH (IR Task)](#mimic-search-ir-task)
- [Model Scripts](#model-scripts)
  - [QA Task Inference](#mimic-ask---inference)
  - [IR Task Inference](#mimic-search---inference)




## Dataset Creation

### MIMICSQL Dataset Generation

1. Download the [MIMIC-III database](https://physionet.org/content/mimiciii/1.4/) (requires PhysioNet authorization) 

2. Generate the [MIMIC SQL dataset](https://github.com/wangpinggl/TREQS/tree/master)

   ```bash
   mkdir mimicsql_db
   python data/create_mimicsql.py --data_dir /path/to/your/MIMIC_III
   ```

   This script will create serialized versions of patient data (text and HTML formats) in the `mimicsql_db` directory.


3. For the `sgen` serialization method, run:
   ```bash
   python data/serialize_sgen.py
   ```
   
   
### MIMIC ASK (QA Task)

1. Generate the MIMIC_ASK dataset:
   ```bash
   python data/create_mimic_ask.py
   ```

   This will generate answers for questions in MIMIC SQL using the previously generated .db files and serializations. 
   
   The data will be stores at `data/mimic_ask` folder.


2. Prepare different prompt variations as described in the paper for a given Feature (all by default):
   ```bash
   python data/serialize_mimic_ask.py
   ```

   This creates variations including:
   - Text and HTML serialization
   - Guidance or non-guidance instructions
   - In-context learning (ICL) versions

3. To apply the `sgen` method, update the serialized corpus path and serialization methods from the script `data/serialize_mimic_ask.py`. Similarly, to apply the `avg` feature, the variables `SERIALIZED_CORPUS` and `PREFIX_CORPUS` should be updated.



### MIMIC SEARCH (IR Task)

1. The queries and qrels are available in the `data/mimic_search` folder.

2. Create the search index (example for using All features):
   ```bash
   python data/index_mimic_search.py --input ./mimicsql_db/list_serialized_corpus.json --output-dir runs --feature all --topk 10
   ```

3. Create the datasets:
   ```bash
   python data/create_mimic_search.py --configuration all_text --b 0.4 --k 0.9
   ```

## Model Scripts

### MIMIC ASK - Inference

Run LLM inference for the Patient Data Extraction task:

```bash
python qa_llm_inference.py --input <input_file> --model <model_name>
```

Options:
- `--input`: Path to the input file with prompts
- `--model`: Name of the LLM model to use for inference

### MIMIC Search - Inference

Run LLM inference for the Patient Retrieval Task:

```bash
python ir_llm_inference.py --input <input_file> --model <model_name> --max-length <max_length> --output_file <output_path> --metrics_file <metrics_output_path>
```

Options:
- `--input`: Path to the input file with prompts
- `--model`: Name of the LLM model to use for inference
- `--max-length`: Context length to use
- `--output_file`: File to save the output information from the model
- `--metrics_file`: File to save the computed metrics




