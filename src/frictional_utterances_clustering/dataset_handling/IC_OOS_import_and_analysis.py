# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd

from frictional_utterances_clustering.dataset_handling.dataset_utils import *

IC_OOS_dataset = load_dict_from_json("data/IC_OOS/data_full.json")

IC_OOS_dataset = IC_OOS_dataset['train'] + IC_OOS_dataset['val'] + IC_OOS_dataset['test']

IC_OOS_counter = 0

utterances_dict = [] 
# utterances = snips_dataset['text'].tolist()
# utterances_embeddings = get_sentence_embeddings(
#     utterances, 'base_language_models/paraphrase-multilingual-mpnet-base-v2')
    
for utterance in IC_OOS_dataset:
    utterances_dict.append({
            'dataset': 'IC_OOS',
            'utterance_id':f'IC_OOS_{IC_OOS_counter}',
            'utterance_split': 'to_define',
            'utterance_lang':'EN',
            'utterance_text':utterance[0],
            'utterance_intent':utterance[1],
            # 'utterance_embedding':utterances_embeddings[index].tolist(),
        })
    IC_OOS_counter += 1

saving_path = f"data/Processed_Datasets/IC_OOS/full_dataset.jsonl"

from_list_of_dict_to_jsonl(saving_path, utterances_dict)