# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from frictional_utterances_clustering.dataset_handling.dataset_utils import *
import pandas as pd
import numpy as np
import os

language_models = []

for language_model in os.listdir('base_language_models/'):
    if '.gitkeep' in language_model: continue
    language_models.append(os.path.join('base_language_models/', language_model))
for language_model in os.listdir('fine_tuned_language_models/'):
    if '.gitkeep' in language_model: continue
    language_models.append(os.path.join('fine_tuned_language_models/', language_model))

datasets_intents_and_languages = load_dict_from_json('data/Datasets_Analysis/datasets_intents_and_languages.json')

for language_model in language_models:
    
#     if (os.path.exists(f"data/Datasets_Analysis/dataset_statistics_by_language_model/{language_model.split('/')[-1]}.json")
#         and 'base_language_models' not in language_model):
#         continue
    
    datasets_infra_intent_avg_similarities = load_dict_from_json(
        f"data/Datasets_Analysis/datasets_infra_intent_avg_similarities/{language_model.split('/')[-1]}.json")

    # 'Snips', 'pathicKnowledge', 'IC_OOS', 'DSTC11', 
    # 'MultiAtis', 'Massive', 'BANKING77', 'CLINC150', 'HWU64'
    datasets = os.listdir('data/Processed_Datasets/') #['IC_OOS', 'pathicKnowledge']

    dataset_statistics = {}

    for dataset in ['DSTC11', 'BANKING77', 'HWU64', 'pathicKnowledge', 'CLINC150', 'Massive']: #datasets
        
        if dataset not in language_model and 'base_language_models' not in language_model:
            continue
        
        if (dataset == 'pathicKnowledge' or dataset == 'Massive') and 'sentence-transformers_all-mpnet-base-v2' in language_model:
                continue
        
        dataset_statistics.setdefault(dataset, {})
        
        intents = datasets_intents_and_languages[dataset]['intents']
        train_intents = datasets_intents_and_languages[dataset]['train_intents']
        dev_intents = datasets_intents_and_languages[dataset]['dev_intents']
        test_intents = datasets_intents_and_languages[dataset]['test_intents']
        languages = datasets_intents_and_languages[dataset]['languages']

        dataset_statistics[dataset]['total_languages'] = len(languages)
        dataset_statistics[dataset]['total_intents'] = len(intents) 
        dataset_statistics[dataset]['train_intents'] = len(train_intents) 
        dataset_statistics[dataset]['dev_intents'] = len(dev_intents) 
        dataset_statistics[dataset]['test_intents'] = len(test_intents) 

        dataset_dataframe = pd.read_csv(f'data/Datasets_Analysis/{dataset}_lang_intent_statistics.csv') 

        intents = dataset_dataframe['utterance_intent'].to_list()
        number_of_utterances_intent_per_lang_array = dataset_dataframe.drop(columns=['utterance_intent']).to_numpy()

        number_of_utterances_per_intent = list(np.nansum(number_of_utterances_intent_per_lang_array, axis=1))

        intent_num_utterances = dict(zip(intents, number_of_utterances_per_intent))

        for key in datasets_infra_intent_avg_similarities[dataset]['overall_statistics']:
            dataset_statistics[dataset][key] = datasets_infra_intent_avg_similarities[dataset]['overall_statistics'][key]
            
        counters = {}

        def update_counter(dataset: str, split: str, key: str):
            global counters
            global intent_num_utterances
            global datasets_infra_intent_avg_similarities
            counters[split].setdefault('num_intents', 0)
            counters[split].setdefault('num_utterances', 0)
            counters[split].setdefault('total', 0)
            counters[split].setdefault('weighted_total', 0)
            counters[split]['num_intents'] += 1
            counters[split]['num_utterances'] += intent_num_utterances[key]
            counters[split]['total'] += datasets_infra_intent_avg_similarities[dataset]['per_intent_statistics'][key]
            counters[split]['weighted_total'] += datasets_infra_intent_avg_similarities[
                dataset]['per_intent_statistics'][key]*intent_num_utterances[key]
        
        for key in datasets_infra_intent_avg_similarities[dataset]['per_intent_statistics'].keys():

            if (key in intent_num_utterances.keys() 
            and not isinstance(datasets_infra_intent_avg_similarities[dataset]['per_intent_statistics'][key], str)):
                counters.setdefault('total', {})
                update_counter(dataset, 'total', key)
                if key in train_intents:
                    counters.setdefault('train', {})
                    update_counter(dataset, 'train', key)
                elif key in dev_intents:
                    counters.setdefault('dev', {})
                    update_counter(dataset, 'dev', key)
                elif key in test_intents:
                    counters.setdefault('test', {})
                    update_counter(dataset, 'test', key)

        for split in ['total', 'train', 'dev', 'test']:
            dataset_statistics[dataset][
                f'{split}_utterances'] = int(counters[split]['num_utterances'])
            dataset_statistics[dataset][
                f'{split}_average_pairwise_similarity_within_intent'] = counters[split]['total']/counters[split]['num_intents']
            dataset_statistics[dataset][
                f'{split}_weighted_average_pairwise_similarity_within_intent'] = counters[split]['weighted_total']/counters[split]['num_utterances']

        save_dict_as_json(dataset_statistics, f"data/Datasets_Analysis/dataset_statistics_by_language_model/{language_model.split('/')[-1]}.json")

