# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd

from frictional_utterances_clustering.dataset_handling.dataset_utils import *

snips_records = import_jsonl_as_list_of_dict('data/Snips/train.jsonl')

snips_dataset = pd.DataFrame(snips_records)

snips_counter = 0

utterances_dict = [] 
# utterances = snips_dataset['text'].tolist()
# utterances_embeddings = get_sentence_embeddings(
#     utterances, 'languagbase_language_modelse_models/paraphrase-multilingual-mpnet-base-v2')
    
for index, row in snips_dataset.iterrows():
    utterances_dict.append({
            'dataset': 'snips',
            'utterance_id':f'snips_{snips_counter}',
            'utterance_split': 'to_define',
            'utterance_lang':'EN',
            'utterance_text':row['text'],
            'utterance_intent':row['label_encoding'],
            # 'utterance_embedding':utterances_embeddings[index].tolist(),
        })
    snips_counter += 1

saving_path = f"data/Processed_Datasets/Snips/full_dataset.jsonl"

from_list_of_dict_to_jsonl(saving_path, utterances_dict)