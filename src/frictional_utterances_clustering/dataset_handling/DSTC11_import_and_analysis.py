# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd

from frictional_utterances_clustering.dataset_handling.dataset_utils import *

DSTC11_dialogues = import_jsonl_as_list_of_dict('data/DSTC11/dialogues.jsonl')
DSTC11_utterances = import_jsonl_as_list_of_dict('data/DSTC11/test-utterances.jsonl')

DSTC11_counter = 0

utterances_dict = [] 
    
for dialoge in DSTC11_dialogues:
    for turn in dialoge["turns"]:
        if len(turn["intents"]) == 1:
            utterances_dict.append({
                    'dataset': 'DSTC11',
                    'utterance_id':f'DSTC11_{DSTC11_counter}',
                    'utterance_split': 'to_define',
                    'utterance_lang':'EN',
                    'utterance_text':turn["utterance"],
                    'utterance_intent':turn["intents"][0],
                    # 'utterance_embedding':utterances_embeddings[index].tolist(),
                })
            DSTC11_counter += 1

for utterance in DSTC11_utterances:
    utterances_dict.append({
            'dataset': 'DSTC11',
            'utterance_id':f'DSTC11_{DSTC11_counter}',
            'utterance_split': 'to_define',
            'utterance_lang':'EN',
            'utterance_text':utterance["utterance"],
            'utterance_intent':utterance['intent'],
            # 'utterance_embedding':utterances_embeddings[index].tolist(),
        })
    DSTC11_counter += 1

saving_path = f"data/Processed_Datasets/DSTC11/full_dataset.jsonl"

from_list_of_dict_to_jsonl(saving_path, utterances_dict)