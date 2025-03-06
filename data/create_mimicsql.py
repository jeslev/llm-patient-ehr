"""
Script to generate the MIMICSQL dataset

Code and CSV file from https://github.com/wangpinggl/TREQS/tree/master/evaluation/process_mimic_db

"""

import os
import re
import csv
import json
import sys
import shutil
import sqlite3
import pandas
import random
import numpy as np
from datetime import datetime
from functools import partial
import copy
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import time



# Configuration
# Specify the path to the downloaded MIMIC III data
data_dir = '/PATH/TO/YOUR/MIMIC'
MIMIC_DB_DIR = 'mimicsql_db'
FILES = ["DEMOGRAPHIC", "DIAGNOSES", "LAB", "PRESCRIPTIONS", "PROCEDURES"]
RANDOM_SEED = 42



# Dictionary mapping for human-readable keys
dictionary_keys = {
    "NAME": "Name", "MARITAL_STATUS": "Marital status", "AGE": "Age", 
    "DOB": "Date of Birth", "GENDER": "Gender", "LANGUAGE": "Language",
    "RELIGION": "Religion", "ADMISSION_TYPE": "Admission type", 
    "DAYS_STAY": "Days stay", "INSURANCE": "Insurance", "ETHNICITY": "Ethnicity",
    "ADMISSION_LOCATION": "Admission location", "DISCHARGE_LOCATION": "Discharge location", 
    "DIAGNOSIS": "Diagnosis", "DOD": "Deathtime",
    "DOB_YEAR": "Year birth", "DOD_YEAR": "Year death", 
    "ADMITTIME": "Admission time", "DISCHTIME": "Discharge time", 
    "ADMITYEAR": "Admission year",
    "ICD9_CODE": "ICD9", "LONG_TITLE": "Long title"
}




def get_patient_name(data_dir):
    pat_id2name = {}
    file_ = os.path.join(data_dir, 'id2name.csv')
    fp = open(file_, 'r')
    for line in csv.reader(fp, delimiter=','):
        pat_id2name[line[0]] = line[1]
    fp.close()
    return pat_id2name


def read_table(data_dir, data_file):
    out_info = []
    
    file_ = os.path.join(data_dir, data_file)
    fp = open(file_, 'r')
    reader = csv.reader(fp, delimiter=',')
    for line in reader:
        header = line
        break
    for line in reader:
        arr = {}
        for k in range(len(header)):
            arr[header[k]] = line[k]
        out_info.append(arr)
    fp.close()
    return out_info


def show_progress(a, b):
    cc = int(round(100.0*float(a)/float(b)))
    dstr = '[' + '>'*cc + ' '*(100-cc) + ']'
    sys.stdout.write(dstr + str(cc) + '%' +'\r')
    sys.stdout.flush()
    

def build_demographic_table(data_dir, out_dir, conn):
    print('Build demographic_table')
    pat_id2name = get_patient_name('data')
    pat_info = read_table(data_dir, 'PATIENTS.csv')
    adm_info = read_table(data_dir, 'ADMISSIONS.csv')
    print('-- Process PATIENTS')
    cnt = 0
    for itm in pat_info:
        cnt += 1
        show_progress(cnt, len(pat_info))
        itm['NAME'] = pat_id2name[itm['SUBJECT_ID']]
        
        dob = datetime.strptime(itm['DOB'], '%Y-%m-%d %H:%M:%S')
        itm['DOB_YEAR'] = str(dob.year)
        
        if len(itm['DOD']) > 0:
            dod = datetime.strptime(itm['DOD'], '%Y-%m-%d %H:%M:%S')
            itm['DOD_YEAR'] = str(dod.year)
        else:
            itm['DOD_YEAR'] = ''
            
    pat_dic = {ky['SUBJECT_ID']: ky for ky in pat_info}
    print()
    print('-- Process ADMISSIONS')
    cnt = 0
    for itm in adm_info:
        cnt += 1
        show_progress(cnt, len(adm_info))
        # patients.csv
        for ss in pat_dic[itm['SUBJECT_ID']]:
            if ss == 'ROW_ID' or ss == 'SUBJECT_ID':
                continue
            itm[ss] = pat_dic[itm['SUBJECT_ID']][ss]
        # admissions.csv
        admtime = datetime.strptime(itm['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        itm['ADMITYEAR'] = str(admtime.year)
        dctime = datetime.strptime(itm['DISCHTIME'], '%Y-%m-%d %H:%M:%S')
        itm['DAYS_STAY'] = str((dctime-admtime).days)
        itm['AGE'] = str(int(itm['ADMITYEAR'])-int(itm['DOB_YEAR']))
        if int(itm['AGE']) > 89:
            itm['AGE'] = str(89+int(itm['AGE'])-300)
    print()
    print('-- write table')
    header = [
        'SUBJECT_ID',
        'HADM_ID',
        'NAME',
        'MARITAL_STATUS',
        'AGE',
        'DOB',
        'GENDER',
        'LANGUAGE',
        'RELIGION',
        
        'ADMISSION_TYPE',
        'DAYS_STAY',
        'INSURANCE',
        'ETHNICITY',
        'EXPIRE_FLAG',
        'ADMISSION_LOCATION',
        'DISCHARGE_LOCATION',
        'DIAGNOSIS',
        
        'DOD',
        'DOB_YEAR',
        'DOD_YEAR',
        
        'ADMITTIME',
        'DISCHTIME',
        'ADMITYEAR'
    ]
            
    fout = open(os.path.join(out_dir,'DEMOGRAPHIC.csv'), 'w')    
    fout.write('\"'+'\",\"'.join(header)+'\"\n')
    for itm in adm_info:
        arr = []
        for wd in header:
            arr.append(itm[wd])
        fout.write('\"'+'\",\"'.join(arr)+'\"\n')
    fout.close()
    print('-- write sql')
    data = pandas.read_csv(
        os.path.join(out_dir,'DEMOGRAPHIC.csv'), 
        dtype={'HADM_ID': str, "DOD_YEAR": float, "SUBJECT_ID": str})
    data.to_sql('DEMOGRAPHIC', conn, if_exists='replace', index=False)


def build_diagnoses_table(data_dir, out_dir, conn):
    print('Build diagnoses_table')
    left = pandas.read_csv(os.path.join(data_dir, 'DIAGNOSES_ICD.csv'), dtype=str)
    right = pandas.read_csv(os.path.join(data_dir, 'D_ICD_DIAGNOSES.csv'), dtype=str)
    left = left.drop(columns=['ROW_ID', 'SEQ_NUM'])
    right = right.drop(columns=['ROW_ID'])
    out = pandas.merge(left, right, on='ICD9_CODE')
    out = out.sort_values(by='HADM_ID')
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'DIAGNOSES.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('DIAGNOSES', conn, if_exists='replace', index=False)


def build_procedures_table(data_dir, out_dir, conn):
    print('Build procedures_table')
    left = pandas.read_csv(os.path.join(data_dir, 'PROCEDURES_ICD.csv'), dtype=str)
    right = pandas.read_csv(os.path.join(data_dir, 'D_ICD_PROCEDURES.csv'), dtype=str)
    left = left.drop(columns=['ROW_ID', 'SEQ_NUM'])
    right = right.drop(columns=['ROW_ID'])
    out = pandas.merge(left, right, on='ICD9_CODE')
    out = out.sort_values(by='HADM_ID')
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'PROCEDURES.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('PROCEDURES', conn, if_exists='replace', index=False) 


def build_prescriptions_table(data_dir, out_dir, conn):
    print('Build prescriptions_table')
    data = pandas.read_csv(os.path.join(data_dir, 'PRESCRIPTIONS.csv'), dtype=str)
    data = data.drop(columns=['ROW_ID', 'GSN', 'DRUG_NAME_POE', 
                              'DRUG_NAME_GENERIC', 'NDC', 'PROD_STRENGTH', 
                              'FORM_VAL_DISP', 'FORM_UNIT_DISP', 
                              'STARTDATE', 'ENDDATE'])
    data = data.dropna(subset=['DOSE_VAL_RX', 'DOSE_UNIT_RX'])
    data['DRUG_DOSE'] = data[['DOSE_VAL_RX', 'DOSE_UNIT_RX']].apply(lambda x: ''.join(x), axis=1)
    data = data.drop(columns=['DOSE_VAL_RX', 'DOSE_UNIT_RX'])
    print('-- write table')
    data.to_csv(os.path.join(out_dir, 'PRESCRIPTIONS.csv'), sep=',', index=False)
    print('-- write sql')
    data.to_sql('PRESCRIPTIONS', conn, if_exists='replace', index=False) 


def build_lab_table(data_dir, out_dir, conn):
    print('Build lab_table')
    cnt = 0
    show_progress(cnt, 4)
    left = pandas.read_csv(os.path.join(data_dir, 'LABEVENTS.csv'), dtype=str)
    cnt += 1
    show_progress(cnt, 4)
    right = pandas.read_csv(os.path.join(data_dir, 'D_LABITEMS.csv'), dtype=str)
    cnt += 1
    show_progress(cnt, 4)
    left = left.dropna(subset=['HADM_ID', 'VALUE', 'VALUEUOM'])
    left = left.drop(columns=['ROW_ID', 'VALUENUM'])
    left['VALUE_UNIT'] = left[['VALUE', 'VALUEUOM']].apply(lambda x: ''.join(x), axis=1)
    left = left.drop(columns=['VALUE', 'VALUEUOM'])
    right = right.drop(columns=['ROW_ID', 'LOINC_CODE'])
    cnt += 1
    show_progress(cnt, 4)
    out = pandas.merge(left, right, on='ITEMID')
    cnt += 1
    show_progress(cnt, 4)
    print()
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'LAB.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('LAB', conn, if_exists='replace', index=False)
    

def create_database_from_MIMIC():

    # Generate five tables and the database with all admissions
    if os.path.exists(MIMIC_DB_DIR):
        shutil.rmtree(MIMIC_DB_DIR)
    os.mkdir(MIMIC_DB_DIR)
    conn = sqlite3.connect(os.path.join(MIMIC_DB_DIR, 'mimic.db'))
    build_demographic_table(data_dir, MIMIC_DB_DIR, conn)
    build_diagnoses_table(data_dir, MIMIC_DB_DIR, conn)
    build_procedures_table(data_dir, MIMIC_DB_DIR, conn)
    build_prescriptions_table(data_dir, MIMIC_DB_DIR, conn)
    build_lab_table(data_dir, MIMIC_DB_DIR, conn)

    '''
    1. We did not emumerate all possible questions about MIMIC III.
    MIMICSQL data is generated based on the patient information 
    related to 100 randomly selected admissions.
    2. The following codes are used for sampling the admissions 
    from the large database. 
    3. The parameter 'random_state=0' in line 41 will provide you 
    the same set of sampled admissions and the same database as we used.
    '''

    print('Begin sampling ...')
    # DEMOGRAPHIC
    print('Processing DEMOGRAPHIC')
    conn = sqlite3.connect(os.path.join(MIMIC_DB_DIR, 'mimic.db'))
    data_demo = pandas.read_csv(os.path.join(MIMIC_DB_DIR, "DEMOGRAPHIC.csv"))
    data_demo_sample = data_demo.sample(5000, random_state=0)
    data_demo_sample = data_demo_sample
    data_demo_sample.to_sql('DEMOGRAPHIC', conn, if_exists='replace', index=False)
    sampled_id = data_demo_sample['HADM_ID'].values

    # DIAGNOSES
    print('Processing DIAGNOSES')
    data_input = pandas.read_csv(os.path.join(MIMIC_DB_DIR, "DIAGNOSES.csv"))
    data_filter = []
    cnt = 0
    for itm in sampled_id:
        msg = 'HADM_ID=='+str(itm)
        data_filter.append(data_input.query(msg))
        cnt += 1
        show_progress(cnt, len(sampled_id))
    data_out = pandas.concat(data_filter, ignore_index=True)
    data_out.to_sql('DIAGNOSES', conn, if_exists='replace', index=False)

    # PROCEDURES
    print('Processing PROCEDURES')
    data_input = pandas.read_csv(os.path.join(MIMIC_DB_DIR, "PROCEDURES.csv"))
    data_filter = []
    cnt = 0
    for itm in sampled_id:
        msg = 'HADM_ID=='+str(itm)
        data_filter.append(data_input.query(msg))
        cnt += 1
        show_progress(cnt, len(sampled_id))
    data_out = pandas.concat(data_filter, ignore_index=True)
    data_out.to_sql('PROCEDURES', conn, if_exists='replace', index=False)

    # PRESCRIPTIONS
    print('Processing PRESCRIPTIONS')
    data_input = pandas.read_csv(os.path.join(MIMIC_DB_DIR, "PRESCRIPTIONS.csv"))
    data_filter = []
    cnt = 0
    for itm in sampled_id:
        msg = 'HADM_ID=='+str(itm)
        data_filter.append(data_input.query(msg))
        cnt += 1
        show_progress(cnt, len(sampled_id))
    data_out = pandas.concat(data_filter, ignore_index=True)
    data_out.to_sql('PRESCRIPTIONS', conn, if_exists='replace', index=False)

    # LAB
    print('Processing LAB')
    data_input = pandas.read_csv(os.path.join(MIMIC_DB_DIR, "LAB.csv"))
    data_filter = []
    cnt = 0
    for itm in sampled_id:
        msg = 'HADM_ID=='+str(itm)
        data_filter.append(data_input.query(msg))
        cnt += 1
        show_progress(cnt, len(sampled_id))
    data_out = pandas.concat(data_filter, ignore_index=True)
    data_out.to_sql('LAB', conn, if_exists='replace', index=False)
    print('Done!')



def load_mimic_data(db_path: str, files: List[str]) -> Tuple[Dict, Dict]:
    """
    Load data from MIMIC CSV files and create dictionary structures.
    
    Args:
        db_path: Path to the MIMIC database files
        files: List of file names to process
    
    Returns:
        Tuple containing tables dictionary and subject to admission mapping
    """
    print("Loading MIMIC data files...")
    
    # Initialize dictionaries
    tables_dict = {file: dict() for file in files}
    subject_to_hadm_id = dict()
    
    for file_name in files:
        print(f"Processing file: {file_name}")
        file_path = os.path.join(db_path, f"{file_name}.csv")
        
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for line in tqdm(reader):
                subject_id = line.pop("SUBJECT_ID")
                hadm_id = line.pop('HADM_ID')
                
                # Track admission time for demographic records
                adm_time = line.get("ADMITTIME", "") if file_name == "DEMOGRAPHIC" else ""
                
                # Process different tables based on their structure
                if file_name == "LAB":
                    process_lab_record(tables_dict, file_name, hadm_id, line)
                elif file_name == "PRESCRIPTIONS":
                    process_prescription_record(tables_dict, file_name, hadm_id, line)
                else:
                    # Handle tables with simpler structures
                    if hadm_id in tables_dict[file_name]:
                        tables_dict[file_name][hadm_id].append(line)
                    else:
                        tables_dict[file_name][hadm_id] = [line]
                
                # Update subject to admission mapping
                update_subject_admission_mapping(subject_to_hadm_id, subject_id, hadm_id, adm_time)
    
    # Print summary statistics
    total_patients = len(subject_to_hadm_id)
    total_admissions = sum(len(admissions) for admissions in subject_to_hadm_id.values())
    print(f"Total number of subjects: {total_patients}, with average admissions per patient: {total_admissions/total_patients:.2f}")
    
    return tables_dict, subject_to_hadm_id



def process_lab_record(tables_dict: Dict, file_name: str, hadm_id: str, line: Dict):
    """Process a laboratory record and update the tables dictionary."""
    item_id = line.pop("ITEMID")
    
    # Convert single values to lists for potential multiple values
    line["CHARTTIME"] = [line["CHARTTIME"]]
    line["VALUE_UNIT"] = [line["VALUE_UNIT"]]
    line["FLAG"] = [line["FLAG"]]
    
    if hadm_id in tables_dict[file_name]:
        if item_id in tables_dict[file_name][hadm_id]:
            # Update existing lab item with additional values
            for label in ["CHARTTIME", "VALUE_UNIT", "FLAG"]:
                if tables_dict[file_name][hadm_id][item_id][label]:
                    tables_dict[file_name][hadm_id][item_id][label].append(line[label][0])
                else:
                    tables_dict[file_name][hadm_id][item_id][label].append("Unspecified")
        else:
            # Add new lab item
            tables_dict[file_name][hadm_id][item_id] = line
    else:
        # Create new admission entry with this lab item
        tables_dict[file_name][hadm_id] = {item_id: line}



def update_subject_admission_mapping(subject_to_hadm_id: Dict, subject_id: str, hadm_id: str, adm_time: str):
    """Update the mapping between subjects and their hospital admissions."""
    if subject_id in subject_to_hadm_id:
        # Check if this admission is already recorded for this subject
        if not any(hadm_id in admission for admission in subject_to_hadm_id[subject_id]):
            subject_to_hadm_id[subject_id].append((hadm_id, adm_time))
    else:
        # Create new subject entry with this admission
        subject_to_hadm_id[subject_id] = [(hadm_id, adm_time)]


def create_subject_data(tables_dict: Dict, subject_to_hadm_id: Dict) -> Dict:
    """
    Create comprehensive subject data dictionary from the tables dictionary.
    
    Args:
        tables_dict: Dictionary containing data from all tables
        subject_to_hadm_id: Mapping between subjects and their admissions
    
    Returns:
        Dictionary with subject data organized by admission
    """
    print("Creating comprehensive subject data...")
    subject_data = {}
    
    for subject_id, admissions in subject_to_hadm_id.items():
        subject_data[subject_id] = []
        
        for hadm_id, adm_time in admissions:
            # Create admission record
            admission_record = {"h_adm_id": hadm_id}
            
            # Add data from each table
            for table in tables_dict.keys():
                if hadm_id in tables_dict[table]:
                    admission_record[table] = tables_dict[table][hadm_id]
                else:
                    admission_record[table] = "Unspecified"
            
            subject_data[subject_id].append(admission_record)
    
    return subject_data


def save_data(data: Dict, filename: str, directory: str = MIMIC_DB_DIR):
    """Save dictionary data as JSON file."""
    filepath = os.path.join(directory, filename)
    print(f"Saving data to {filepath}")
    
    with open(filepath, 'w') as fp:
        json.dump(data, fp)


def create_filtered_subject_data(subject_data: Dict, mapping: Dict) -> Dict:
    """
    Filter subject data to include only specific admissions.
    
    Args:
        subject_data: Original subject data
        mapping: Dictionary mapping patient IDs to admission IDs
    
    Returns:
        Filtered subject data dictionary
    """
    filtered_data = {}
    
    for patient_id, admissions in subject_data.items():
        if patient_id in mapping:
            target_adm_id = mapping[patient_id]
            for admission in admissions:
                if admission['h_adm_id'] == target_adm_id:
                    filtered_data[patient_id] = [admission]
                    break
    
    return filtered_data


def random_pick_values(person_data: List[Dict], percentage: float = 0.6) -> List[Dict]:
    """
    Create a reduced profile by randomly selecting a subset of values.
    
    Args:
        person_data: List containing one admission record for a person
        percentage: Percentage of values to keep (0.0 to 1.0)
    
    Returns:
        List containing one reduced admission record
    """
    if not person_data:
        return []
    
    # Work with the first admission
    admission = person_data[0]
    reduced_profile = {'h_adm_id': admission['h_adm_id']}
    
    # Process DEMOGRAPHIC table
    demos = admission['DEMOGRAPHIC']
    new_demos = []
    for demo in demos:
        original_keys = list(demo.keys())
        # Always keep the NAME field and randomly select other fields
        selected_keys = random.sample(original_keys, int(len(original_keys) * 0.2))
        filtered_demo = {'NAME': demo.get('NAME', 'Unknown')}
        for key in selected_keys:
            filtered_demo[key] = demo[key]
        new_demos.append(filtered_demo)
    reduced_profile['DEMOGRAPHIC'] = new_demos
    
    # Process DIAGNOSES table
    diagnoses = admission['DIAGNOSES']
    if diagnoses == 'Unspecified':
        reduced_profile['DIAGNOSES'] = diagnoses
    else:
        selected_diags = random.sample(diagnoses, int(len(diagnoses) * percentage))
        reduced_profile['DIAGNOSES'] = selected_diags
    
    # Process LAB table
    labs = admission['LAB']
    if labs == 'Unspecified':
        reduced_profile['LAB'] = labs
    else:
        new_labs = {}
        original_keys = list(labs.keys())
        selected_keys = random.sample(original_keys, int(len(original_keys) * percentage))
        for key in selected_keys:
            new_labs[key] = labs[key]
        reduced_profile['LAB'] = new_labs
    
    # Process PRESCRIPTIONS table
    prescriptions = admission['PRESCRIPTIONS']
    if prescriptions == 'Unspecified':
        reduced_profile['PRESCRIPTIONS'] = prescriptions
    else:
        new_prescriptions = {}
        original_keys = list(prescriptions.keys())
        selected_keys = random.sample(original_keys, int(len(original_keys) * percentage))
        for key in selected_keys:
            new_prescriptions[key] = prescriptions[key]
        reduced_profile['PRESCRIPTIONS'] = new_prescriptions
    
    # Process PROCEDURES table
    procedures = admission['PROCEDURES']
    if procedures == 'Unspecified':
        reduced_profile['PROCEDURES'] = procedures
    else:
        selected_procedures = random.sample(procedures, int(len(procedures) * percentage))
        reduced_profile['PROCEDURES'] = selected_procedures
    
    return [reduced_profile]


def process_prescription_record(tables_dict: Dict, file_name: str, hadm_id: str, line: Dict):
    """Process a prescription record and update the tables dictionary."""
    drug = line.pop("DRUG")
    
    # Convert single values to lists for potential multiple values
    line["DRUG_DOSE"] = [line["DRUG_DOSE"]]
    
    if hadm_id in tables_dict[file_name]:
        if drug in tables_dict[file_name][hadm_id]:
            # Update existing drug with additional doses
            tables_dict[file_name][hadm_id][drug]["DRUG_DOSE"].append(line["DRUG_DOSE"][0])
        else:
            # Add new drug
            tables_dict[file_name][hadm_id][drug] = line
    else:
        # Create new admission entry with this drug
        tables_dict[file_name][hadm_id] = {drug: line}



def create_random_features_corpus(filtered_data: Dict, reduction_percentage: float) -> Dict:
    """
    Create a corpus with randomly reduced features for each patient.
    
    Args:
        filtered_data: Filtered subject data
        reduction_percentage: Percentage of features to keep (0.0 to 1.0)
    
    Returns:
        Dictionary with randomly reduced features
    """
    print(f"Creating random features corpus with {reduction_percentage:.1%} feature retention...")
    random.seed(RANDOM_SEED)
    
    random_features_corpus = {}
    for patient_id, admission_data in filtered_data.items():
        random_features_corpus[patient_id] = random_pick_values(admission_data, reduction_percentage)
    
    return random_features_corpus


# Template definitions
def create_templates(use_list_aggregation=False):
    """
    Create and return all templates used for serialization.
    
    Args:
        use_list_aggregation: If True, use ALL aggregation for lists instead of MEAN
    
    Returns:
        Dictionary of templates
    """
    
    # Template for procedure data
    template_procedure = {
        "start": {
            "default": "The procedures performed are: ", 
            "singular": "The procedure performed is: ", 
            "empty": "No procedures recorded. "
        },
        'template_text': "#LONG_TITLE# (#SHORT_TITLE#) with ICD9 code #ICD9_CODE#; ",
        "template_data": {}
    }

    # Template for prescription data
    template_prescriptions = {
        "start": {
            "default": "The administered medication amounts are: ", 
            "singular": "The administered medication amount is: ", 
            "empty": "No medications recorded. "
        },
        'template_text': "#CODE# #DRUG_DOSE# through #ROUTE# route during ICU stay #ICUSTAY_ID#, with #DRUG_TYPE# type and code as #FORMULARY_DRUG_CD#; ",
        'template_data': {
            "DRUG_DOSE": (True, "ALL" if use_list_aggregation else "MEAN"),
            "DRUG_TYPE": (False, "LOWER")
        }
    }

    # Template for diagnosis data
    template_diagnoses = {
        "start": {
            "default": "The diagnoses are: ", 
            "singular": "The diagnosis is: ", 
            "empty": "No diagnoses recorded."
        },
        'template_text': "#LONG_TITLE# (#SHORT_TITLE#) with ICD9 code #ICD9_CODE#; ",
        'template_data': {}
    }

    # Template for demographic data
    template_demographic = {
        "start": {"default": ""},
        'template_text': "named #NAME#, #AGE# years old, #GENDER#, with ethnicity as #ETHNICITY#, language as #LANGUAGE#, "
                        "marital status as #MARITAL_STATUS#, religion as #RELIGION#, insured under #INSURANCE#. "
                        "The patient was admitted for #DAYS_STAY# days, from #ADMITTIME# to #DISCHTIME# "
                        "in #ADMISSION_LOCATION#. The patient was discharged at #DISCHARGE_LOCATION#. "
                        "The admission is #ADMISSION_TYPE# type and the main disease is #DIAGNOSIS#. ",
        'template_data': {
            "ETHNICITY": (False, "LOWER"),
            "LANGUAGE": (False, "LOWER"),
            "RELIGION": (False, "LOWER"),
            "ADMITTIME": (False, "DATE"),
            "DISCHTIME": (False, "DATE"),
            "MARITAL_STATUS": (False, "LOWER"),
            "ADMISSION_LOCATION": (False, "LOWER"),
            "DISCHARGE_LOCATION": (False, "LOWER"),
            "ADMISSION_TYPE": (False, "LOWER"),
            "DIAGNOSIS": (False, "LOWER"),
            "DOD": (False, "DATE"),
            "GENDER": (False, "GENDER")
        }
    }

    # Template for lab data
    template_lab = {
        "start": {
            "default": "The lab tests measured are: ",
            "singular": "The lab test measured is: ",
            "empty": "No lab tests recorded. "
        },
        'template_text': "#VALUE_UNIT# #LABEL# in #FLUID# fluid, categorized under #CATEGORY#, and flagged as #FLAG#; ",
        'template_data': {
            "VALUE_UNIT": (True, "ALL" if use_list_aggregation else "MEAN"),
            "FLUID": (False, "LOWER"),
            "FLAG": (False, "ALL_UNIQUE"),
            "CHARTTIME": (False, "FIRST_LAST")
        }
    }

    # Combine templates for admission
    template_admission = {
        'PROCEDURES': template_procedure,
        'PRESCRIPTIONS': template_prescriptions,
        'DIAGNOSES': template_diagnoses,
        'DEMOGRAPHIC': template_demographic,
        'LAB': template_lab,
        'template_text': "#DEMOGRAPHIC# #DIAGNOSES# #LAB# #PROCEDURES# #PRESCRIPTIONS#"
    }

    # Template for specific subject
    template_subject_specific = {
        "start": {"default": "Patient ID: #ID#, "}, 
        "admission": template_admission, 
        "separator": "\n"
    }
    
    return template_subject_specific


# HTML template definitions
def create_html_templates(use_list_aggregation=False):
    """
    Create templates for HTML generation.
    
    Args:
        use_list_aggregation: If True, use ALL aggregation for lists instead of MEAN
        
    Returns:
        Dictionary of HTML templates
    """
    # HTML template for procedure data
    template_procedure = {
        "start": {"default": ""},
        'template_text': "#LONG_TITLE# with ICD9 code #ICD9_CODE#",
        "template_data": {}
    }

    # HTML template for prescription data
    template_prescriptions = {
        "start": {"default": ""},
        'template_text': "#CODE# #DRUG_DOSE#, route: #ROUTE#, ICU stay: #ICUSTAY_ID#, type: #DRUG_TYPE# code: #FORMULARY_DRUG_CD#",
        'template_data': {
            "DRUG_DOSE": (True, "ALL" if use_list_aggregation else "MEAN"),
            "DRUG_TYPE": (False, "LOWER")
        }
    }

    # HTML template for diagnosis data
    template_diagnoses = {
        "start": {"default": ""},
        'template_text': "#LONG_TITLE# with ICD9 code #ICD9_CODE#",
        'template_data': {}
    }

    # HTML template for demographic data
    template_demographic = {
        "start": {"default": ""},
        'template_text': "",
        'template_data': {
            "ETHNICITY": (False, "LOWER"),
            "LANGUAGE": (False, "LOWER"),
            "RELIGION": (False, "LOWER"),
            "ADMITTIME": (False, "DATE"),
            "DISCHTIME": (False, "DATE"),
            "MARITAL_STATUS": (False, "LOWER"),
            "ADMISSION_LOCATION": (False, "LOWER"),
            "DISCHARGE_LOCATION": (False, "LOWER"),
            "ADMISSION_TYPE": (False, "LOWER"),
            "DIAGNOSIS": (False, "LOWER"),
            "DOD": (False, "DATE"),
            "GENDER": (False, "GENDER")
        }
    }

    # HTML template for lab data
    template_lab = {
        "start": {"default": ""},
        'template_text': "#VALUE_UNIT#, fluid: #FLUID#, category: #CATEGORY#, flag: #FLAG#, code #CODE#",
        'template_data': {
            "VALUE_UNIT": (True, "ALL" if use_list_aggregation else "MEAN"),
            "FLUID": (False, "LOWER"),
            "FLAG": (False, "ALL_UNIQUE"),
            "CHARTTIME": (False, "FIRST_LAST")
        }
    }

    # Combine HTML templates
    template_data_html = {
        'DEMOGRAPHIC': template_demographic,
        'LAB': template_lab,
        'PROCEDURES': template_procedure,
        'PRESCRIPTIONS': template_prescriptions,
        'DIAGNOSES': template_diagnoses
    }
    
    return template_data_html


# Define mappings for specific patient admissions
def get_subject_admission_mapping():
    """Return mapping of subject IDs to their specific admission IDs."""
    return {2560: 126863, 6983: 173511, 81923: 101866, 29961: 196409, 3343: 193452, 62296: 193576, 15898: 163908, 7273: 151733,
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


# Data transformation functions
def preprocess_simple_dict(d: Dict) -> Dict:
    """Replace empty or None values with 'unspecified'."""
    for key in d.keys():
        if d[key] == '' or d[key] is None:
            d[key] = "unspecified"
    return d

def dict_to_list_dict(d: Dict) -> List[Dict]:
    """Convert dictionary of dictionaries to list of dictionaries with CODE key."""
    result = []
    for key in d.keys():
        temp_dict = dict()
        temp_dict.update(d[key])
        temp_dict.update({"CODE": key})
        result.append(temp_dict)
    return result

def separate_number_unit(value: str) -> Tuple[float, Optional[str]]:
    """Separate numeric value and unit from a string."""
    try:
        numeric = '0123456789-.'
        for i, c in enumerate(value):
            if c not in numeric:
                break
        number = float(value[:i])
        unit = value[i:].lstrip()
        return (number, unit)
    except:
        return (0, None)

def date_to_text(date_time: str) -> str:
    """Convert date-time string to readable text format."""
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    try:
        (date, time) = date_time.split(" ")
        (year, month, day) = date.split("-")
        (hour, minutes, seconds) = time.split(":")    
        return f"{months[int(month) - 1]} {day} {year} at {hour}h{minutes}"
    except:
        return date_time

def replace_placeholder(match, data, template_data):
    """Replace placeholder in text with actual data value."""
    key = match.group(1)  # Get text between #
    value = data.get(key, match.group(0))
    
    contains_unit, aggregate_mode = template_data.get(key, (False, False))
    
    if aggregate_mode:
        return aggregate_values(value, mode=aggregate_mode, contains_unit=contains_unit)
    
    if isinstance(value, list):
        return aggregate_values(value, mode="ALL", contains_unit=contains_unit)
    
    return value


def aggregate_values(value_to_aggregate: Union[str, List], mode: str, contains_unit: bool = False) -> str:
    """
    Aggregate values based on specified mode.
    
    Args:
        value_to_aggregate: Value(s) to aggregate
        mode: Aggregation mode (DATE, FIRST_LAST, LOWER, GENDER, ALL, ALL_UNIQUE, MEAN)
        contains_unit: Whether values contain units that should be processed separately
        
    Returns:
        Aggregated string representation
    """
    # Handle simple string case
    if isinstance(value_to_aggregate, str):
        if mode == "DATE":
            return date_to_text(value_to_aggregate)
        elif mode == "LOWER":
            return value_to_aggregate.lower()
        elif mode == "GENDER":
            gender_dict = {"M": 'male', "F": 'female'}
            return gender_dict.get(value_to_aggregate.upper(), "")
        else:
            return value_to_aggregate
    
    # Handle empty list case
    if not value_to_aggregate:
        return "unspecified"
    
    # Handle different aggregation modes for lists
    if mode == "DATE":
        try:
            return date_to_text(value_to_aggregate)
        except:
            return value_to_aggregate
            
    elif mode == "FIRST_LAST":
        if len(value_to_aggregate) < 2:
            all_values = ", ".join(value_to_aggregate)
            return all_values if all_values else "unspecified"
        else:
            return f"from {value_to_aggregate[0]} to {value_to_aggregate[-1]}"
            
    elif mode == "ALL":
        all_values = ", ".join(value_to_aggregate)
        return all_values if all_values else "unspecified"
            
    elif mode == "ALL_UNIQUE":
        unique_values = ", ".join(v for v in set(value_to_aggregate) if v)
        return unique_values if unique_values else "unspecified"
            
    elif mode == "MEAN" and contains_unit:
        values_with_units = [separate_number_unit(i) for i in value_to_aggregate]
        values_list = [i[0] for i in values_with_units if i[1] is not None]
        units_list = [i[1] for i in values_with_units if i[1] is not None]
        
        if not values_list:
            return "unspecified"
            
        units = list(set(units_list))
        result_parts = []
        
        for unit in units:
            unit_values = [i[0] for i in values_with_units if i[1] == unit]
            if unit_values:
                mean = np.round(np.mean(np.array(unit_values)), 3)
                result_parts.append(f"{mean} {unit}")
                
        return " or ".join(result_parts) if result_parts else "unspecified"
            
    elif mode == "MEAN" and not contains_unit:
        try:
            numeric_values = [float(i) for i in value_to_aggregate]
            mean = np.round(np.mean(np.array(numeric_values)))
            return str(mean)
        except:
            return "unspecified"
            
    else:
        return ", ".join(value_to_aggregate) if value_to_aggregate else "unspecified"

def replace_placeholder(match, data, template_data):
    """Replace placeholder in text with actual data value."""
    key = match.group(1)  # Get text between #
    value = data.get(key, match.group(0))
    
    contains_unit, aggregate_mode = template_data.get(key, (False, False))
    
    if aggregate_mode:
        return aggregate_values(value, mode=aggregate_mode, contains_unit=contains_unit)
    
    if isinstance(value, list):
        return aggregate_values(value, mode="ALL", contains_unit=contains_unit)
    
    return value

def get_template_dict(data: Dict, template_text: str, template_data: Dict) -> str:
    """Fill template with data values."""
    partial_replace = partial(replace_placeholder, data=data, template_data=template_data)
    pattern = re.compile(r"#(\w+)#")
    text = pattern.sub(partial_replace, template_text)
    return text

def get_template_table(data: Union[Dict, List, str], template: Dict) -> str:
    """Format table data according to template."""
    # Convert data to list of dictionaries if needed
    if not isinstance(data, list):
        if isinstance(data, dict):
            data = dict_to_list_dict(data)
        else:
            data = []
    
    # Handle empty data case
    if len(data) == 0:
        return template.get("start").get("empty", template.get("start").get("default", ""))
    
    # Get appropriate start text based on data size
    if len(data) == 1:
        start = template.get("start").get("singular", template.get("start").get("default", ""))
    else:
        start = template.get("start").get("default", "")
    
    # Process each item
    for item in data:
        preprocess_simple_dict(item)
        start += get_template_dict(item, template['template_text'], template['template_data'])
    
    # Remove trailing separator and add period
    return start[:-2] + "."

def get_template_admission(admission: Dict, template: Dict) -> str:
    """Format admission data according to template."""
    text = template['template_text']
    
    for key in ['DEMOGRAPHIC', 'PROCEDURES', 'PRESCRIPTIONS', "LAB", "DIAGNOSES"]:
        text_table = get_template_table(admission[key], template[key])
        text = text.replace(f"#{key}#", text_table)
    
    return text

def get_template_subject(admissions: List[Dict], template: Dict, subject_id: str) -> str:
    """Format subject's admissions data according to template."""
    if len(admissions) == 0:
        return template.get("start").get("empty", template.get("start").get("default", ""))
    
    if len(admissions) == 1:
        start = template.get("start").get("singular", template.get("start").get("default", ""))
    else:
        start = template.get("start").get("default", "")
    
    start = start.replace("#ID#", subject_id)
    
    for admission in admissions:
        text_admission = get_template_admission(admission, template['admission'])
        start += text_admission + template['separator']
    
    return start[:-len(template['separator'])]

# HTML generation functions
def template_html_v2(admission: Dict, template_data: Dict) -> Tuple[str, List[Tuple]]:
    """
    Generate an HTML table for the admission data.
    
    Args:
        admission: Dictionary containing admission data
        template_data: Template data for formatting
        
    Returns:
        Tuple of (HTML table string, list of key-value tuples)
    """
    start = ""
    end = ""
    tuples = []
    
    # Process demographic data
    for demographic in admission['DEMOGRAPHIC']:
        for key in demographic:
            value = demographic[key]
            contains_unit, aggregate_mode = template_data['DEMOGRAPHIC'].get('template_data', {}).get(key, (False, False))
            if aggregate_mode:
                value = aggregate_values(value, mode=aggregate_mode, contains_unit=contains_unit)
            if isinstance(value, list):
                value = aggregate_values(value, mode="ALL", contains_unit=contains_unit)
            tuples.append((dictionary_keys.get(key, key), value))
            start += f"<th>{dictionary_keys.get(key, key)}</th>"
            end += f"<td>{value}</td>"
    
    # Process diagnoses
    if len(admission['DIAGNOSES']) > 0:
        start += f"<th>Diagnoses D</th>"
        end += f"<td></td>"
    for diagnosis in admission['DIAGNOSES']:
        if isinstance(diagnosis, dict):
            head = diagnosis.get('SHORT_TITLE', '')
            template = template_data['DIAGNOSES']
            value = get_template_dict(diagnosis, template['template_text'], template['template_data'])
            start += f"<th>D: {head}</th>"
            end += f"<td>{value}</td>"
            tuples.append((head, value))
    
    # Process lab tests
    if len(admission['LAB']) > 0:
        start += f"<th>Lab L</th>"
        end += f"<td></td>"
    for lab in admission['LAB']:
        if isinstance(lab, dict):
            head = lab.get('LABEL', '')
            template = template_data['LAB']
            value = get_template_dict(lab, template['template_text'], template['template_data'])
            start += f"<th>L: {head}</th>"
            end += f"<td>{value}</td>"
            tuples.append((head, value))
    
    # Process procedures
    if len(admission['PROCEDURES']) > 0:
        start += f"<th>Procedure P</th>"
        end += f"<td></td>"
    for procedure in admission['PROCEDURES']:
        if isinstance(procedure, dict):
            head = procedure.get('SHORT_TITLE', '')
            template = template_data['PROCEDURES']
            value = get_template_dict(procedure, template['template_text'], template['template_data'])
            start += f"<th>P: {head}</th>"
            end += f"<td>{value}</td>"
            tuples.append((head, value))
    
    # Process prescriptions
    if len(admission['PRESCRIPTIONS']) > 0:
        start += f"<th>Prescription R</th>"
        end += f"<td></td>"
    for prescription in admission['PRESCRIPTIONS']:
        if isinstance(prescription, dict):
            head = prescription.get('CODE', '')
            template = template_data['PRESCRIPTIONS']
            value = get_template_dict(prescription, template['template_text'], template['template_data'])
            start += f"<th>R: {head}</th>"
            end += f"<td>{value}</td>"
            tuples.append((head, value))
    
    # Construct and return the HTML table
    html_table = f"<table border=\"1\"><thead><tr>{start}</tr></thead><tbody><tr>{end}</tr></tbody></table>"
    return html_table, tuples

def template_html_grouped(admission: Dict, template_data: Dict, group_size: int) -> str:
    """
    Generate an HTML table with grouped data for the admission.
    
    Args:
        admission: Dictionary containing admission data
        template_data: Template data for formatting
        group_size: Number of items to group together in a single column
        
    Returns:
        HTML table string
    """
    start = ""
    end = ""
    heads = []
    values = []
    
    # Process demographic data
    for demographic in admission['DEMOGRAPHIC']:
        for key in demographic:
            value = demographic[key]
            contains_unit, aggregate_mode = template_data['DEMOGRAPHIC'].get('template_data', {}).get(key, (False, False))
            if aggregate_mode:
                value = aggregate_values(value, mode=aggregate_mode, contains_unit=contains_unit)
            if isinstance(value, list):
                value = aggregate_values(value, mode="ALL", contains_unit=contains_unit)
            
            start += f"<th>{dictionary_keys.get(key, key)}</th>"
            end += f"<td>{value}</td>"
    
    # Process diagnoses in groups
    if len(admission['DIAGNOSES']) > 0:
        start += f"<th>Diagnoses D</th>"
        end += f"<td></td>"
    
    diagnoses_list = sorted([elem for elem in admission['DIAGNOSES'] if isinstance(elem, dict)], 
                           key=lambda x: x.get('SHORT_TITLE', ""))
    
    for i, diagnosis in enumerate(diagnoses_list):
        head = diagnosis.get('SHORT_TITLE', '')
        template = template_data['DIAGNOSES']
        value = get_template_dict(diagnosis, template['template_text'], template['template_data'])
        
        heads.append(head)
        values.append(value)
        if (i+1) % group_size == 0 or i == len(diagnoses_list) - 1:
            head_group = " <> ".join(heads)
            value_group = " <> ".join(values)
            start += f"<th>D: {head_group}</th>"
            end += f"<td>{value_group}</td>"
            heads = []
            values = []
    
    # Process lab tests in groups
    if len(admission['LAB']) > 0:
        start += f"<th>Lab L</th>"
        end += f"<td></td>"
    
    lab_list = sorted([elem for elem in admission['LAB'] if isinstance(elem, dict)], 
                     key=lambda x: x.get("LABEL", ""))
    
    for i, lab in enumerate(lab_list):
        head = lab.get('LABEL', '')
        template = template_data['LAB']
        value = get_template_dict(lab, template['template_text'], template['template_data'])
        
        heads.append(head)
        values.append(value)
        if (i+1) % group_size == 0 or i == len(lab_list) - 1:
            head_group = " <> ".join(heads)
            value_group = " <> ".join(values)
            start += f"<th>L: {head_group}</th>"
            end += f"<td>{value_group}</td>"
            heads = []
            values = []
    
    # Process procedures in groups
    if len(admission['PROCEDURES']) > 0:
        start += f"<th>Procedure P</th>"
        end += f"<td></td>"
    
    procedures_list = sorted([elem for elem in admission['PROCEDURES'] if isinstance(elem, dict)], 
                            key=lambda x: x.get("SHORT_TITLE", ""))
    
    for i, procedure in enumerate(procedures_list):
        head = procedure.get('SHORT_TITLE', '')
        template = template_data['PROCEDURES']
        value = get_template_dict(procedure, template['template_text'], template['template_data'])
        
        heads.append(head)
        values.append(value)
        if (i+1) % group_size == 0 or i == len(procedures_list) - 1:
            head_group = " <> ".join(heads)
            value_group = " <> ".join(values)
            start += f"<th>P: {head_group}</th>"
            end += f"<td>{value_group}</td>"
            heads = []
            values = []
    
    # Process prescriptions in groups
    if len(admission['PRESCRIPTIONS']) > 0:
        start += f"<th>Prescription R</th>"
        end += f"<td></td>"
    
    prescriptions_list = sorted([elem for elem in admission['PRESCRIPTIONS'] if isinstance(elem, dict)], 
                               key=lambda x: x.get("CODE", ""))
    
    for i, prescription in enumerate(prescriptions_list):
        head = prescription.get('CODE', '')
        template = template_data['PRESCRIPTIONS']
        value = get_template_dict(prescription, template['template_text'], template['template_data'])
        
        heads.append(head)
        values.append(value)
        if (i+1) % group_size == 0 or i == len(prescriptions_list) - 1:
            head_group = " <> ".join(heads)
            value_group = " <> ".join(values)
            start += f"<th>R: {head_group}</th>"
            end += f"<td>{value_group}</td>"
            heads = []
            values = []
    
    # Construct and return the HTML table
    return f"<table border=\"1\"><thead><tr>{start}</tr></thead><tbody><tr>{end}</tr></tbody></table>"

def count_corpus_elements(data: Dict) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Count occurrences of elements in the corpus.
    
    Args:
        data: Dictionary of patient data
        
    Returns:
        Tuple of dictionaries counting LAB, DIAGNOSES, PRESCRIPTIONS, and PROCEDURES elements
    """
    lab_counts = {}
    diagnoses_counts = {}
    prescriptions_counts = {}
    procedures_counts = {}
    
    for subject in data:
        for admission in data[subject]:
            # Count LAB elements
            if not isinstance(admission['LAB'], str):
                for record_id in admission['LAB']:
                    lab_counts[record_id] = lab_counts.get(record_id, 0) + 1
            
            # Count DIAGNOSES elements
            if not isinstance(admission['DIAGNOSES'], str):
                for diagnosis in admission['DIAGNOSES']:
                    code = diagnosis['ICD9_CODE']
                    diagnoses_counts[code] = diagnoses_counts.get(code, 0) + 1
            
            # Count PRESCRIPTIONS elements
            if not isinstance(admission['PRESCRIPTIONS'], str):
                for record_id in admission['PRESCRIPTIONS']:
                    prescriptions_counts[record_id] = prescriptions_counts.get(record_id, 0) + 1
            
            # Count PROCEDURES elements
            if not isinstance(admission['PROCEDURES'], str):
                for procedure in admission['PROCEDURES']:
                    code = procedure['ICD9_CODE']
                    procedures_counts[code] = procedures_counts.get(code, 0) + 1
    
    return lab_counts, diagnoses_counts, prescriptions_counts, procedures_counts

def process_admission(admission: Dict, frequency_counts: Dict[str, Dict] = None) -> Dict:
    """
    Process admission data, limiting entries to top-k most frequent elements.
    
    Args:
        admission: Admission data dictionary
        frequency_counts: Dictionary of frequency counts for each element type
    
    Returns:
        Processed admission with limited entries
    """
    # Define limits for each category
    category_limits = {
        'LAB': 10, 
        'PROCEDURES': 5, 
        'DIAGNOSES': 15, 
        'PRESCRIPTIONS': 20
    }
    
    # Create a new dictionary for the processed admission
    admission_data_limited = {
        'h_adm_id': admission['h_adm_id'],
        'DEMOGRAPHIC': admission['DEMOGRAPHIC']
    }
    
    # If frequency counts not provided, don't sort by frequency
    if not frequency_counts:
        frequency_counts = {
            'lab': {},
            'diagnoses': {},
            'prescriptions': {},
            'procedures': {}
        }
    
    lab_counts = frequency_counts.get('lab', {})
    diagnoses_counts = frequency_counts.get('diagnoses', {})
    prescriptions_counts = frequency_counts.get('prescriptions', {})
    procedures_counts = frequency_counts.get('procedures', {})
    
    # Process PRESCRIPTIONS - take top 20
    if isinstance(admission['PRESCRIPTIONS'], dict):
        prescriptions_list = copy.deepcopy(dict_to_list_dict(admission['PRESCRIPTIONS']))
        prescriptions_list.sort(key=lambda row: prescriptions_counts.get(row['CODE'], 0), reverse=True)
        admission_data_limited['PRESCRIPTIONS'] = prescriptions_list[:category_limits['PRESCRIPTIONS']]
    else:
        admission_data_limited['PRESCRIPTIONS'] = admission['PRESCRIPTIONS']
    
    # Process LAB - take top 10
    if isinstance(admission['LAB'], dict):
        lab_list = copy.deepcopy(dict_to_list_dict(admission['LAB']))
        lab_list.sort(key=lambda row: lab_counts.get(row['CODE'], 0), reverse=True)
        admission_data_limited['LAB'] = lab_list[:category_limits['LAB']]
    else:
        admission_data_limited['LAB'] = admission['LAB']
    
    # Process PROCEDURES - take top 5
    if isinstance(admission['PROCEDURES'], list):
        procedures_list = copy.deepcopy(admission['PROCEDURES'])
        procedures_list.sort(key=lambda r: procedures_counts.get(r['ICD9_CODE'], 0), reverse=True)
        admission_data_limited['PROCEDURES'] = procedures_list[:category_limits['PROCEDURES']]
    else:
        admission_data_limited['PROCEDURES'] = admission['PROCEDURES']
    
    # Process DIAGNOSES - take top 15
    if isinstance(admission['DIAGNOSES'], list):
        diagnoses_list = copy.deepcopy(admission['DIAGNOSES'])
        diagnoses_list.sort(key=lambda r: diagnoses_counts.get(r['ICD9_CODE'], 0), reverse=True)
        admission_data_limited['DIAGNOSES'] = diagnoses_list[:category_limits['DIAGNOSES']]
    else:
        admission_data_limited['DIAGNOSES'] = admission['DIAGNOSES']
    
    return admission_data_limited

def create_serialized_corpus(input_file: str, output_file: str, include_html: bool = True, checkpoint_interval: int = 10000,
                             use_list_aggregation = False):
    """
    Create serialized text and HTML representations of clinical data.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        include_html: Whether to include HTML representations
        checkpoint_interval: Number of records to process before saving checkpoint
    """
    # Load regular text templates
    text_template = create_templates(use_list_aggregation=use_list_aggregation)
    
    # Load HTML templates if needed
    html_templates = create_html_templates(use_list_aggregation=use_list_aggregation)
    
    # Load subject-to-admission mapping
    subj_adm_mapping = get_subject_admission_mapping()
    
    # Load input data
    print(f"Loading input data from {input_file}")
    with open(input_file) as json_file:
        table_in_json = json.load(json_file)
    
    print(f"Processing {len(table_in_json)} patient records")
    
    # Count element frequencies
    print("Counting element frequencies...")
    frequency_counts = {
        'lab': {},
        'diagnoses': {},
        'prescriptions': {},
        'procedures': {}
    }
    
    frequency_counts['lab'], frequency_counts['diagnoses'], frequency_counts['prescriptions'], frequency_counts['procedures'] = count_corpus_elements(table_in_json)
    
    # Create serialized corpus
    print("Creating serialized corpus...")
    data_serialized = {}
    count = 0
    
    for subject_id in tqdm(table_in_json.keys()):
        count += 1
        
        # Save checkpoint periodically
        if count % checkpoint_interval == 0:
            print(f"Checkpoint at {count} records")
            with open(output_file, 'w') as fp:
                json.dump(data_serialized, fp)
        
        admissions = table_in_json[subject_id]
        
        # Determine which admission to use
        if int(subject_id) in subj_adm_mapping:
            # Find specific admission used for questions
            target_hadm_id = str(subj_adm_mapping[int(subject_id)])
            admission = next((adm for adm in admissions if str(adm['h_adm_id']) == target_hadm_id), admissions[-1])
        else:
            # Use last admission as default
            admission = admissions[-1]
        
        # Process admission
        processed_admission = process_admission(admission, frequency_counts)
        
        # Generate text representation
        serialized_text = get_template_subject([processed_admission], text_template, subject_id)
        
        # Store result with basic text template
        data_serialized[subject_id] = {
            "h_adm_id": admission["h_adm_id"],
            "template": serialized_text
        }
        
        # Generate HTML representations if requested
        if include_html:
            # Generate detailed HTML table
            html_short, tuples = template_html_v2(processed_admission, html_templates)
            data_serialized[subject_id]["html_short"] = html_short
            data_serialized[subject_id]["tuples"] = tuples
            
            # Generate grouped HTML tables with different group sizes
            data_serialized[subject_id]["html_short_g2"] = template_html_grouped(processed_admission, html_templates, 2)
            #data_serialized[subject_id]["html_short_g3"] = template_html_grouped(processed_admission, html_templates, 3)
            #data_serialized[subject_id]["html_short_g4"] = template_html_grouped(processed_admission, html_templates, 4)
    
    # Save final output
    print(f"Saving final output to {output_file}")
    with open(output_file, 'w') as fp:
        json.dump(data_serialized, fp)
    
    print(f"Processing complete. {count} records serialized.")


def gather_and_serializate_data():
    """Main function for the data processing workflow."""


    os.makedirs(MIMIC_DB_DIR, exist_ok=True)
    
    
    # Load data from MIMIC CSV files
    tables_dict, subject_to_hadm_id = load_mimic_data(MIMIC_DB_DIR, FILES)
    
    # Create comprehensive subject data
    subject_data = create_subject_data(tables_dict, subject_to_hadm_id)
    
    # Save complete data
    save_data(subject_data, 'subject_data_rep.json')
    save_data(subject_to_hadm_id, 'subject_to_hadm_id_rep.json')
    
    # Load serialized corpus to filter admissions
    with open(os.path.join(MIMIC_DB_DIR, 'subject_data_rep.json'), 'r') as fp:
        rand_subject_data = json.load(fp)
    
    # Assuming ser_data is available (need to load this or modify the code)
    # For demonstration purposes, create a simple mapping
    try:
        with open(f'{MIMIC_DB_DIR}/serialized_data.json', 'r') as fp:
            ser_data = json.load(fp)
        
        map_pat_adm = {k: v['h_adm_id'] for k, v in ser_data.items()}
    except FileNotFoundError:
        # For demonstration, use the first admission of each patient
        print("Warning: serialized_data.json not found, using first admission for each patient")
        map_pat_adm = {k: v[0]['h_adm_id'] for k, v in subject_data.items() if v}
    
    # Filter subject data
    filtered_subject_data = create_filtered_subject_data(rand_subject_data, map_pat_adm)
    
    # Create random features corpus with 20% feature retention
    random_features_corpus = create_random_features_corpus(filtered_subject_data, 0.2)
    
    # Save random features corpus
    save_data(random_features_corpus, 'rand_feat_subject_data_rep40.json')
    
    print("Data processing completed successfully.")





def get_table_names(conn: sqlite3.Connection) -> List[str]:
    """
    Get the list of all tables in the database.
    
    Args:
        conn: Database connection
        
    Returns:
        List of table names
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]


def get_table_create_statement(conn: sqlite3.Connection, table_name: str) -> str:
    """
    Get the CREATE TABLE statement for a specific table.
    
    Args:
        conn: Database connection
        table_name: Name of the table
        
    Returns:
        CREATE TABLE statement
    """
    cursor = conn.cursor()
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    create_statement = cursor.fetchone()[0]
    return create_statement


def get_distinct_subject_ids(conn: sqlite3.Connection) -> List[int]:
    """
    Get all distinct SUBJECT_ID values from the DEMOGRAPHIC table.
    
    Args:
        conn: Database connection
        
    Returns:
        List of distinct SUBJECT_ID values
    """
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT SUBJECT_ID FROM DEMOGRAPHIC;")
    subject_ids = cursor.fetchall()
    return [int(row[0]) for row in subject_ids]


def get_filtered_demographic_data(conn: sqlite3.Connection) -> List[Tuple]:
    """
    Get filtered DEMOGRAPHIC data based on the specified criteria.
    
    Args:
        conn: Database connection
        
    Returns:
        List of filtered DEMOGRAPHIC rows
    """
    # Get distinct subject IDs
    subject_ids = get_distinct_subject_ids(conn)
    
    # Get all column names for the DEMOGRAPHIC table
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(DEMOGRAPHIC);")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    # Build the query for retrieving data
    columns_str = ", ".join(column_names)
    
    # For each subject ID, get the appropriate admission(s)
    filtered_rows = []
    
    print(f"Processing {len(subject_ids)} distinct subject IDs...")
    for subject_id in tqdm(subject_ids):
        if subject_id in get_subject_admission_mapping():
            # Get the specific admission for this subject
            hadm_id = get_subject_admission_mapping()[subject_id]
            cursor.execute(f"SELECT {columns_str} FROM DEMOGRAPHIC WHERE SUBJECT_ID = ? AND HADM_ID = ?", 
                          (subject_id, hadm_id))
            rows = cursor.fetchall()
            
            if rows:
                filtered_rows.extend(rows)
            else:
                # If specified admission not found, fall back to most recent
                print(f"Specified admission {hadm_id} for subject {subject_id} not found, using most recent instead")
                cursor.execute(f"SELECT {columns_str} FROM DEMOGRAPHIC WHERE SUBJECT_ID = ? ORDER BY ADMITTIME DESC LIMIT 1", 
                              (subject_id,))
                rows = cursor.fetchall()
                filtered_rows.extend(rows)
        else:
            # Get the most recent admission for this subject
            cursor.execute(f"SELECT {columns_str} FROM DEMOGRAPHIC WHERE SUBJECT_ID = ? ORDER BY ADMITTIME DESC LIMIT 1", 
                          (subject_id,))
            rows = cursor.fetchall()
            filtered_rows.extend(rows)
    
    return filtered_rows


def copy_table_data(source_conn: sqlite3.Connection, target_conn: sqlite3.Connection, 
                   table_name: str, batch_size: int = 1000) -> None:
    """
    Copy data from one table to another, possibly with filtering.
    
    Args:
        source_conn: Source database connection
        target_conn: Target database connection
        table_name: Name of the table to copy
        batch_size: Batch size for insert operations
    """
    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()
    
    # Get column names
    source_cursor.execute(f"PRAGMA table_info({table_name});")
    columns = source_cursor.fetchall()
    column_names = [col[1] for col in columns]
    columns_str = ", ".join(column_names)
    placeholders = ", ".join(["?" for _ in column_names])
    
    # Handle DEMOGRAPHIC table specially
    if table_name == "DEMOGRAPHIC":
        filtered_rows = get_filtered_demographic_data(source_conn)
        
        # Insert filtered data in batches
        print(f"Inserting {len(filtered_rows)} filtered DEMOGRAPHIC rows in batches...")
        for i in range(0, len(filtered_rows), batch_size):
            batch = filtered_rows[i:i+batch_size]
            target_cursor.executemany(f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})", batch)
            target_conn.commit()
            
        print(f"Inserted {len(filtered_rows)} filtered DEMOGRAPHIC rows")
    else:
        # For other tables, copy all data in batches
        source_cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        total_rows = source_cursor.fetchone()[0]
        
        source_cursor.execute(f"SELECT {columns_str} FROM {table_name};")
        
        # Prepare batches for insertion
        print(f"Copying {total_rows} rows from {table_name} in batches...")
        batch = source_cursor.fetchmany(batch_size)
        
        while batch:
            target_cursor.executemany(f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})", batch)
            target_conn.commit()
            batch = source_cursor.fetchmany(batch_size)


def copy_database_structure(source_conn: sqlite3.Connection, target_conn: sqlite3.Connection) -> None:
    """
    Copy the database structure (all tables) from source to target.
    
    Args:
        source_conn: Source database connection
        target_conn: Target database connection
    """
    tables = get_table_names(source_conn)
    
    # Copy table definitions
    for table_name in tables:
        create_statement = get_table_create_statement(source_conn, table_name)
        target_conn.execute(create_statement)
    
    target_conn.commit()
    print(f"Copied structure of {len(tables)} tables")


def create_single_admission_db(INPUT_DB_PATH='./mimicsql_db/mimic.db', OUTPUT_DB_PATH='./mimicsql_db/mimic_qa.db') -> None:
    """
    Main function to filter the database.
    """
    # Check if input database exists
    if not os.path.exists(INPUT_DB_PATH):
        raise FileNotFoundError(f"Input database {INPUT_DB_PATH} not found")
    
    # Remove output database if it exists
    if os.path.exists(OUTPUT_DB_PATH):
        os.remove(OUTPUT_DB_PATH)
    
    print(f"Filtering database from {INPUT_DB_PATH} to {OUTPUT_DB_PATH}")
    
    # Connect to both databases
    source_conn = sqlite3.connect(INPUT_DB_PATH)
    target_conn = sqlite3.connect(OUTPUT_DB_PATH)
    
    try:
        # Copy database structure
        print("Copying database structure...")
        copy_database_structure(source_conn, target_conn)
        
        # Copy/filter table data
        tables = get_table_names(source_conn)
        for table_name in tables:
            print(f"\nProcessing table: {table_name}")
            copy_table_data(source_conn, target_conn, table_name)
        
        print("\nDatabase filtering complete!")
        
        # Print statistics
        for table_name in tables:
            source_cursor = source_conn.cursor()
            target_cursor = target_conn.cursor()
            
            source_cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            source_count = source_cursor.fetchone()[0]
            
            target_cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            target_count = target_cursor.fetchone()[0]
            
            print(f"Table {table_name}: {source_count} rows -> {target_count} rows")
    
    finally:
        # Close connections
        source_conn.close()
        target_conn.close()





def main():
    create_database_from_MIMIC()
    
    gather_and_serializate_data()

    # Create serialized corpus with both text and HTML representations
    create_serialized_corpus(
        f'{MIMIC_DB_DIR}/subject_data_rep.json', 
        f'{MIMIC_DB_DIR}/serialized_corpus.json', 
        include_html=True,
        checkpoint_interval=10000,
        use_list_aggregation=False,
    )
    
    # Create a list-based serialized corpus (using ALL aggregation instead of MEAN)
    create_serialized_corpus(
        f'{MIMIC_DB_DIR}/subject_data_rep.json', 
        f'{MIMIC_DB_DIR}/list_serialized_corpus.json', 
        include_html=True,
        checkpoint_interval=10000,
        use_list_aggregation=True,
    )

    # Create special version for QA task
    start_time = time.time()
    create_single_admission_db()
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()