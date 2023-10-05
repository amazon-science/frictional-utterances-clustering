# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd
import glob
import os

from frictional_utterances_clustering.dataset_handling.dataset_utils import *

multiAtis_files_path = "data/MultilingualNLU/data/MultiATIS++/data/train_dev_test"
all_files = glob.glob(multiAtis_files_path + "/*.tsv")
single_language_dataframes = [pd.read_csv(file, on_bad_lines = 'warn', sep='\t') for file in all_files] 

multi_atis_counter = 0

for (name_file, dataset) in zip(all_files, single_language_dataframes):
    
    new_name_file = os.path.basename(os.path.normpath(name_file))

    utterances_dict = [] 
    # utterances = dataset['utterance'].tolist()
    # utterances_embeddings = get_sentence_embeddings(utterances)
    utterance_split = new_name_file[:-7]
    utterance_language = new_name_file[-6:-4]
    
    utterances_dict = []

    for index, row in dataset.iterrows():
        if row['slot_labels']:
            slot_labels = row['slot_labels']
        else:
            slot_labels = row['slot-labels']
        utterances_dict.append({
            'dataset': 'multiatis',
            'utterance_id':f'multiatis_{multi_atis_counter}',
            'utterance_split':utterance_split,
            'utterance_lang':utterance_language,
            'utterance_text':row['utterance'],
            'utterance_slot_labels':slot_labels,
            'utterance_intent':row['intent'],
            # 'utterance_embedding':utterances_embeddings[index].tolist(),
        })
        multi_atis_counter += 1

    saving_path = f"data/Processed_Datasets/MultiAtis/{new_name_file[:-4]}.jsonl"

    from_list_of_dict_to_jsonl(saving_path, utterances_dict)