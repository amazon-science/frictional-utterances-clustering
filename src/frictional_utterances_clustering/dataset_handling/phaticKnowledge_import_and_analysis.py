# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd

from frictional_utterances_clustering.dataset_handling.dataset_utils import *

pathicKnowledge_dataset = pd.read_csv("data/Phatic_Knowledge/phatic_new.csv")

pathicKnowledge_counter = 0

print(pathicKnowledge_dataset)

print(pathicKnowledge_dataset.columns)

utterances_dict = [] 
# utterances = pathicKnowledge_dataset['Utterance Text'].tolist()
# utterances_embeddings = get_sentence_embeddings(utterances)
    
for index, row in pathicKnowledge_dataset.iterrows():
    utterances_dict.append({
            'dataset': 'pathicKnowledge',
            'utterance_id':f'pathicKnowledge_{pathicKnowledge_counter}',
            'utterance_split': 'to_define',
            'utterance_lang':'IT',
            'utterance_text':row['Utterance Text'],
            'utterance_intent':row["SIM"].rsplit('/', 1)[-1],
            # 'utterance_embedding':utterances_embeddings[index].tolist(),
            'utterance_H1_intent':row["H1 Intent"],
            'utterance_H1_domain':row["H1 Domain"],
            'utterance_coverage_type':row["coverage_type"],
            'utterance_nlu_console_segmentation_oracle':row["nlu_console_segmentation_oracle"],

        })
    pathicKnowledge_counter += 1

saving_path = f"data/Processed_Datasets/pathicKnowledge/full_dataset.jsonl"

from_list_of_dict_to_jsonl(saving_path, utterances_dict)