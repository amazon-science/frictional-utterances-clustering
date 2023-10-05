# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd
import glob
import os
from os.path import exists

from frictional_utterances_clustering.dataset_handling.dataset_utils import *

massive_files_path = "data/Massive Amazon/data"
all_files = glob.glob(massive_files_path + "/*.jsonl")

massive_counter = 0

for file_path in all_files:
    
    new_name_file = os.path.basename(os.path.normpath(file_path))
    print(new_name_file)

    if exists(f"data/Processed_Datasets/Massive/full_dataset_{new_name_file[-8:-6]}.jsonl"):
        continue

    massive_records = import_jsonl_as_list_of_dict(file_path)
    massive_dataset = pd.DataFrame(massive_records)

    utterances_dict = [] 
    # utterances = massive_dataset['utt'].tolist()
    # utterances_embeddings = get_sentence_embeddings(utterances)

    for index, row in massive_dataset.iterrows():
        utterances_dict.append({
                'dataset': 'massive',
                'utterance_id':f'massive_{massive_counter}',
                'utterance_split': 'to_define',
                'utterance_lang':new_name_file[-8:-6],
                'utterance_text':row['utt'],
                'utterance_intent':row['intent'],
                'utterance_scenario':row['scenario'],
                'utterance_annot_utt':row['annot_utt'],
                'utterance_partition':row['partition'],
                'utterance_locale':row['locale'],
                #'utterance_judgments':row['judgments'],
                'utterance_worker_id':row['worker_id'],
                #'utterance_slot_method':row['slot_method'],
                # 'utterance_embedding':utterances_embeddings[index].tolist(),
            })
        massive_counter += 1

    saving_path = f"data/Processed_Datasets/Massive/full_dataset_{new_name_file[-8:-6]}.jsonl"

    from_list_of_dict_to_jsonl(saving_path, utterances_dict)