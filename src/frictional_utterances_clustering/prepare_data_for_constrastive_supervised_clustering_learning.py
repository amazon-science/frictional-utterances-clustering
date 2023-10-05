# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd
import os

from frictional_utterances_clustering.utils.experiments_utils import *
from frictional_utterances_clustering.utils.clustering_metrics import *
from frictional_utterances_clustering.utils.clustering_algorithms import *
from frictional_utterances_clustering.dataset_handling.dataset_utils import *

experiment_number = 4

datasets_languages_and_intents = load_dict_from_json(
    f"data/Datasets_Analysis/datasets_intents_and_languages_{experiment_number}.json")

datasets = os.listdir('data/Processed_Datasets/')

dataframes = []

languages_per_dataset = {
    'Snips': [], 
    'pathicKnowledge': [], 
    'MultiAtis': [],
    'Massive': ["AL", "AM", "AZ", "BD", "CN", "DE", "DK", "ES", "ET", "FI", "FR", "GB", "GE", "GR", "HU", "ID", "IL", "IN", "IR", "IS", "IT", "JP", "KE", "KH", "KR", "LV", "MM", "MN", "MY", "NL", "NO", "PH", "PK", "PL", "PT", "RO", "RU", "SA", "SE", "SL", "TH", "TR", "TW", "US", "VN", "ZA"],
    'IC_OOS': [], 
    'DSTC11': [],
    'BANKING77': [],
    'CLINC150': [],  
    'HWU64': []
}

frac_samples_to_use_per_dataset = {
    'Snips': 1.0, 
    'pathicKnowledge': 1.0, 
    'MultiAtis': 0.50,
    'Massive': 0.20,
    'IC_OOS': 1.0, 
    'DSTC11': 1.0,
    'BANKING77': 1.0,
    'CLINC150': 0.75,  
    'HWU64': 1.0
}

for dataset in datasets:
    
    print(dataset)
    
    lang = languages_per_dataset[dataset]

    frac_of_sample_to_use = frac_samples_to_use_per_dataset[dataset]

    experiment_dataset = join_all_jsonl_records_in_single_list(
        f"data/Processed_Datasets/{dataset}", lang, 
        percentage_dataset_to_include=frac_of_sample_to_use)
    
    train_intents = datasets_languages_and_intents[dataset]["train_intents"]
    dev_intents = datasets_languages_and_intents[dataset]["dev_intents"]
    test_intents = datasets_languages_and_intents[dataset]["test_intents"]

    experiment_dataset = pd.DataFrame(experiment_dataset)

    assign_split_to_utterances(experiment_dataset, train_intents, dev_intents, test_intents)

    train_dataset_without_embeddings = experiment_dataset[experiment_dataset['utterance_split'] == 'train']
    dev_dataset_without_embeddings = experiment_dataset[experiment_dataset['utterance_split'] == 'dev']
    test_dataset_without_embeddings = experiment_dataset[experiment_dataset['utterance_split'] == 'test']

    saving_path = f"data/Final_Datasets_For_Experiments/{dataset}/"

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    
    train_dataset_without_embeddings.to_csv(
        os.path.join(saving_path, f'train_{experiment_number}.csv'), index=False)
    dev_dataset_without_embeddings.to_csv(
        os.path.join(saving_path, f'dev_{experiment_number}.csv'), index=False)
    test_dataset_without_embeddings.to_csv(
        os.path.join(saving_path, f'test_{experiment_number}.csv'), index=False)
    
    dataframes.append(experiment_dataset)

final_dataset = pd.concat(dataframes).reset_index(drop=True)

print(final_dataset[[
    'dataset', 'utterance_id', 'utterance_split', 
    'utterance_lang', 'utterance_text', 'utterance_intent']])

train_dataset_without_embeddings = final_dataset[final_dataset['utterance_split'] == 'train']
dev_dataset_without_embeddings = final_dataset[final_dataset['utterance_split'] == 'dev']
test_dataset_without_embeddings = final_dataset[final_dataset['utterance_split'] == 'test']

train_dataset_without_embeddings.to_csv(
    f'data/Final_Datasets_For_Experiments/All_Datasets/train_{experiment_number}.csv', index=False)
dev_dataset_without_embeddings.to_csv(
    f'data/Final_Datasets_For_Experiments/All_Datasets/dev_{experiment_number}.csv', index=False)
test_dataset_without_embeddings.to_csv(
    f'data/Final_Datasets_For_Experiments/All_Datasets/test_{experiment_number}.csv', index=False)

