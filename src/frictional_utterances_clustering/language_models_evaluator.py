# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from frictional_utterances_clustering.dataset_handling.dataset_utils import *
import pandas as pd
import numpy as np
import os

datasets_intents_and_languages = load_dict_from_json(
    'data/Datasets_Analysis/datasets_intents_and_languages.json')

all_datasets_all_language_models_summary_statistics = {}

for dataset in datasets_intents_and_languages.keys():
    
    all_datasets_all_language_models_summary_statistics.setdefault(dataset, {})
    all_datasets_all_language_models_summary_statistics[
        dataset].setdefault('overall_statistics_by_language_model', {})
    all_datasets_all_language_models_summary_statistics[
        dataset].setdefault('intent_similarities_by_language_model', {})
    for split in ['train_intents', 'dev_intents', 'test_intents']:
        all_datasets_all_language_models_summary_statistics[dataset][
                'intent_similarities_by_language_model'].setdefault(split, {})
        for intent in datasets_intents_and_languages[dataset][split]:
            all_datasets_all_language_models_summary_statistics[dataset][
                'intent_similarities_by_language_model'][split].setdefault(intent, {})

dataset_statistics = 'data/Datasets_Analysis/dataset_statistics_by_language_model/'
intent_similarities = 'data/Datasets_Analysis/datasets_infra_intent_avg_similarities/'

dataset_statistics_by_language_model = os.listdir(dataset_statistics)
intent_similarities_by_language_model = os.listdir(intent_similarities)

for language_model in dataset_statistics_by_language_model:
    language_model_name = language_model[:-5]
    try: 
        statistics = load_dict_from_json(os.path.join(dataset_statistics, language_model))
    except: continue
    for dataset in statistics.keys():
        for metric in statistics[dataset].keys():
            if 'intents' not in metric and 'utterances' not in metric:
                all_datasets_all_language_models_summary_statistics[
                    dataset]['overall_statistics_by_language_model'].setdefault(metric, {})
                all_datasets_all_language_models_summary_statistics[
                    dataset]['overall_statistics_by_language_model'][
                    metric][language_model_name]=statistics[dataset][metric]

for language_model in intent_similarities_by_language_model:
    language_model_name = language_model[:-5]
    try:
        similarities = load_dict_from_json(os.path.join(intent_similarities, language_model))
    except: continue
    for dataset in similarities.keys():
        for intent in similarities[dataset]['per_intent_statistics'].keys():
            value = similarities[dataset]['per_intent_statistics'][intent]
            if intent in datasets_intents_and_languages[dataset]['train_intents']:
                all_datasets_all_language_models_summary_statistics[dataset][
                'intent_similarities_by_language_model']['train_intents'][intent][language_model_name] = value
            elif intent in datasets_intents_and_languages[dataset]['dev_intents']:
                all_datasets_all_language_models_summary_statistics[dataset][
                'intent_similarities_by_language_model']['dev_intents'][intent][language_model_name] = value                
            elif intent in datasets_intents_and_languages[dataset]['test_intents']:
                all_datasets_all_language_models_summary_statistics[dataset][
                'intent_similarities_by_language_model']['test_intents'][intent][language_model_name] = value

save_dict_as_json(
    all_datasets_all_language_models_summary_statistics, 
    f"data/Datasets_Analysis/all_datasets_all_language_models_summary_statistics.json")

results_post_processing = []

overall_statistics = all_datasets_all_language_models_summary_statistics

for dataset in overall_statistics.keys():
    print(dataset)
    statistics = overall_statistics[dataset]["overall_statistics_by_language_model"]
    try:
        model_list = statistics[f"train_average_infra_intent_pairwise_similarity"].keys()
    except:
        continue
    for split in ["dev", "test", "train"]:
        for model in model_list:   
            infra_intent = statistics[f"{split}_average_infra_intent_pairwise_similarity"][model]
            within_intent = statistics[f"{split}_average_pairwise_similarity_within_intent"][model]
            acc_at_1 = statistics[f"{split}_retrieval_accuracy_at_1"][model]
            sim_at_1 = statistics[f"{split}_retrieval_similarity_at_1"][model]
            experiment = {
                "model": model,
                "dataset": dataset,
                "split": split,
                "average_infra_intent_pairwise_similarity": infra_intent,
                "average_pairwise_similarity_within_intent": within_intent,
                "retrieval_accuracy_at_1": acc_at_1,
                "retrieval_similarity_at_1": sim_at_1,
            }
            
            results_post_processing.append(experiment)
    
    

df = pd.DataFrame(results_post_processing)

df.to_csv(
    'data/Datasets_Analysis/all_datasets_all_language_models_summary_statistics.txt', 
    decimal=',', 
    float_format='%.3f')

