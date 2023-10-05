# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd
import os

from frictional_utterances_clustering.dataset_handling.dataset_utils import *

for dataset in ['BANKING77', 'CLINC150', 'HWU64']:
    dataset_path = f'data/Few-Shot-Intent-Detection/Datasets/{dataset}'

    dataset_counter = 0
    utterances_dict = [] 
    
    for split in ['train', 'valid', 'test']:

        samples_path = os.path.join(dataset_path, split)

        intents = set()
        
        with open(f'{samples_path}/seq.in', 'r', encoding="utf-8") as f_text, open(f'{samples_path}/label', 'r', encoding="utf-8") as f_label:
            for text, label in zip(f_text, f_label):
                utterances_dict.append({
                    'dataset': dataset,
                    'utterance_id':f'{dataset}_{dataset_counter}',
                    'utterance_split': 'to_define',
                    'utterance_lang':'EN',
                    'utterance_text':text.strip(),
                    'utterance_intent':label.strip(),
                })
                intents.add(label.strip())
                dataset_counter += 1
  
  # Create a new directory because it does not exist 

    saving_path = f"data/Processed_Datasets/{dataset}/"

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    saving_file_path = os.path.join(saving_path, 'full_dataset.jsonl')

    from_list_of_dict_to_jsonl(saving_file_path, utterances_dict)