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

datasets_languages_and_intents = load_dict_from_json(
    "data/Datasets_Analysis/datasets_intents_and_languages.json")

language_models = []

for language_model in os.listdir('base_language_models/'):
    if '.gitkeep' in language_model: continue
    language_models.append(os.path.join('base_language_models/', language_model))
for language_model in os.listdir('fine_tuned_language_models/'):
    if '.gitkeep' in language_model: continue
    language_models.append(os.path.join('fine_tuned_language_models/', language_model))

for language_model in language_models:
    
#     if (os.path.exists(
#         f"data/Datasets_Analysis/datasets_infra_intent_avg_similarities/{language_model.split('/')[-1]}.json")
#         and 'base_language_models' not in language_model):
#         continue
    
    similarities = {}
    
    datasets = os.listdir('data/Processed_Datasets/')

    # 'Snips', 'pathicKnowledge', 'IC_OOS', 'DSTC11', 
    # 'MultiAtis', 'Massive', 'BANKING77', 'CLINC150', 'HWU64'
    for dataset in ['DSTC11', 'BANKING77', 'HWU64', 'pathicKnowledge', 'CLINC150', 'Massive']: #['pathicKnowledge', 'IC_OOS'] or datasets
        
        if dataset not in language_model and 'base_language_models' not in language_model:
            continue
        
        if (dataset == 'pathicKnowledge' or dataset == 'Massive') and 'sentence-transformers_all-mpnet-base-v2' in language_model:
                continue
            
        print(f"DATASET: {dataset}")

        lang = []

        if dataset == 'Massive':
            lang = ['US', "ES", "IT", "DE", "RU", "CN", "FR"]
        # elif dataset == 'MultiAtis':
        #     lang = ["EN", "ES", "FR"]

        if dataset == 'MultiAtis':
            frac_of_sample_to_use = 0.5
        elif dataset == 'Massive':
            frac_of_sample_to_use = 0.10
        else:
            frac_of_sample_to_use = 1.0


        dataset_records = join_all_jsonl_records_in_single_list(
            f"data/Processed_Datasets/{dataset}", lang, percentage_dataset_to_include=frac_of_sample_to_use)

        df = pd.DataFrame(dataset_records)
        
        train_intents = datasets_languages_and_intents[dataset]['train_intents']
        dev_intents = datasets_languages_and_intents[dataset]['dev_intents']
        test_intents = datasets_languages_and_intents[dataset]['test_intents']
        
        assign_split_to_utterances(df, train_intents, dev_intents, test_intents)
        
        similarities[dataset], tSNE_plot = get_average_similarity_per_intent_and_tSNE_plot(
            df, train_intents, dev_intents, test_intents, language_model_path=f'{language_model}')

        saving_path = f"data/Datasets_Analysis/tSNE_plots/{dataset}/"

        if not os.path.exists(saving_path):
            os.makedirs(saving_path)

        tSNE_plot.savefig(
            os.path.join(saving_path, f"{language_model.split('/')[-1]}.png"),
            bbox_inches='tight'
        )

    print("SAVING THE RESULTS")

    save_dict_as_json(similarities, f"data/Datasets_Analysis/datasets_infra_intent_avg_similarities/{language_model.split('/')[-1]}.json")

    print("RESULTS SAVED")
    