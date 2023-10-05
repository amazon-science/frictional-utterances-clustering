# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd

from frictional_utterances_clustering.utils.experiments_utils import *
from frictional_utterances_clustering.utils.clustering_metrics import *
from frictional_utterances_clustering.utils.clustering_algorithms import *
from frictional_utterances_clustering.dataset_handling.dataset_utils import *

datasets_languages_and_intents = load_dict_from_json(
    "data/Datasets_Analysis/datasets_intents_and_languages_with_splits.json")

experiment_category = 'monolingual'

language_models = []

# for language_model in os.listdir('base_language_models/'):
#     if '.gitkeep' in language_model: continue
#     language_models.append(os.path.join('base_language_models/', language_model))
for language_model in os.listdir('fine_tuned_language_models/'):
    if '.gitkeep' in language_model: continue
    language_models.append(os.path.join('fine_tuned_language_models/', language_model))

# monolingual:
# 'pathicKnowledge', ok
# 'IC_OOS', 
# 'DSTC11', 
# 'MultiAtis', 
# 'Massive', 
# 'BANKING77', ok
# 'CLINC150', ok
# 'HWU64', 

datasets = os.listdir('data/Processed_Datasets/')

# adjusted mutual information score: ['DSTC11', 'BANKING77', 'HWU64', 'pathicKnowledge', 'CLINC150','Massive']
# missing k-means: ['CLINC150', 'pathicKnowledge']

datasets = ['Massive'] #['DSTC11', 'HWU64', 'pathicKnowledge', 'Massive', 'CLINC150', 'BANKING77']

total_gold_clusters = {
    'Snips': [2*2], 
    'pathicKnowledge': [2000], 
    'MultiAtis': [250],
    'Massive': [800],
    'IC_OOS': [53*2], 
    'DSTC11': [250], #5*2
    'BANKING77': [800],
    'CLINC150': [1500],  
    'HWU64': [700]
}
interval_steps = {
    'Snips': [2], 
    'pathicKnowledge': [50], 
    'MultiAtis': [5],
    'Massive': [16],
    'IC_OOS': [10], 
    'DSTC11': [5], #2
    'BANKING77': [16],
    'CLINC150': [30],  
    'HWU64': [14]
}

monolingual_lang_dict = {
    'Snips': ['EN'], 
    'pathicKnowledge': ['IT'], 
    'MultiAtis': ["DE", "EN", "ES", "FR", "HI", "JA", "PT", "TR", "ZH"],
    'Massive': ['US', "ES", "IT", "DE", "RU", "CN", "FR"],
    'IC_OOS': ['EN'], 
    'DSTC11': ['EN'],
    'BANKING77': ['EN'],
    'CLINC150': ['EN'],  
    'HWU64': ['EN']
}

monolingual_dict_frac_to_use = {
    'Snips': 1.0, 
    'pathicKnowledge': 1.0, 
    'MultiAtis': 0.50,
    'Massive': 0.50,
    'IC_OOS': 1.0, 
    'DSTC11': 1.0,
    'BANKING77': 1.0,
    'CLINC150': 0.75,  
    'HWU64': 1.0
}

multilingual_dict_frac_to_use = {
    'MultiAtis': 1.0,
    'Massive': 0.10,
    'All_Datasets': 0.10
}

clustering_algorithms = {
    'connected_componentes': connected_components,
    'DBSCAN': DBSCAN,
    #'optimized_k_means': optimized_k_means,
    'agglomerative_hierarchical_clustering': agglomerative_hierarchical_clustering, 
    }
    
parameters_to_optimize = {
    'connected_componentes':  {
        'cut_threshold': [
            0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1, 1.05, 1.1],
    },
    'agglomerative_hierarchical_clustering': {
        'linkage': ['ward', 'complete', 'average'],
        'distance_threshold': [
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1, 
            1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.50, 1.95], # 1.55, 1.6, 1.65, 1.70, 1.75, 1.8, 1.85, 1.90, 
    },
    'DBSCAN': {
        'eps': [
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.6, 0.65, 0.70, 0.75, 0.8, 0.85, 0.90, 0.95, 1], 
        'min_samples': [2, 5, 10, 15, 20, 25, 30],
    },
    'optimized_k_means': {
        'max_clusters':[],
        'interval_step':[]
    }

}

for i in range(4, 5):
    for dataset in datasets:
        
        print(dataset)
        
        parameters_to_optimize[
            'optimized_k_means']['max_clusters'] = total_gold_clusters[dataset]
        parameters_to_optimize[
            'optimized_k_means']['interval_step'] = interval_steps[dataset]
        
        if experiment_category == 'monolingual':
            list_of_languages = monolingual_lang_dict[dataset]
            frac_of_sample_to_use = monolingual_dict_frac_to_use[dataset]
        else:
            list_of_languages = [
            'MY', 'FI', 'JP', 'BD', 'ET', 'IT', 'PH', 'AZ', 'SE', 'JA', 'GE', 
            'IN', 'AL', 'VN', 'KH', 'NO', 'US', 'CN', 'TW', 'ES', 'PK', 'DE', 
            'RO', 'TH', 'ZH', 'MN', 'LV', 'EN', 'IS', 'GB', 'SL', 'HI', 'KR', 
            'PT', 'MM', 'KE', 'NL', 'GR', 'IL', 'FR', 'TR', 'SA', 'DK', 'ZA', 
            'HU', 'PL', 'IR', 'AM', 'RU', 'ID']

            frac_of_sample_to_use = multilingual_dict_frac_to_use[dataset]
        
#         train_dataset = pd.read_csv(
#             f'data/Final_Datasets_For_Experiments/{dataset}/train.csv', 
#             usecols = ['utterance_intent', 'utterance_text', 'utterance_lang'])
        dev_dataset = pd.read_csv(
            f'data/Final_Datasets_For_Experiments/{dataset}/dev.csv', 
            usecols = ['utterance_intent', 'utterance_text', 'utterance_lang'])
        test_dataset = pd.read_csv(
            f'data/Final_Datasets_For_Experiments/{dataset}/test.csv', 
            usecols = ['utterance_intent', 'utterance_text', 'utterance_lang'])

#         train_dataset = train_dataset[
#             train_dataset['utterance_lang'].isin(list_of_languages)].sample(
#                 frac=frac_of_sample_to_use)
        
        dev_dataset = dev_dataset[
            dev_dataset['utterance_lang'].isin(list_of_languages)].sample(
                frac=frac_of_sample_to_use)
        
        test_dataset = test_dataset[
            test_dataset['utterance_lang'].isin(list_of_languages)].sample(
                frac=frac_of_sample_to_use)
        
        for language_model_path in language_models:
            
            print(language_model_path)

            if (dataset == 'pathicKnowledge' or dataset == 'Massive') and 'sentence-transformers_all-mpnet-base-v2' in language_model_path:
                continue
            
            if (dataset not in language_model_path 
                and 'base_language_models' not in language_model_path):
                continue
            if (dataset in language_model_path 
                and experiment_category == 'multilingual' 
                and 'multilingual' not in language_model_path):
                continue
            
            print('EXTRACTING THE FEATURES')
            
#             train_features = prepare_features_for_clustering(
#                 train_dataset, language_model=language_model_to_use)
            dev_features = prepare_features_for_clustering(
                dev_dataset, language_model=language_model_path)
            test_features = prepare_features_for_clustering(
                test_dataset, language_model=language_model_path)
            
            print('EXPERIMENTS ARE STARTING')

            for algorithm in clustering_algorithms.keys():
                for optimization_criterion in ['adjusted_mutual_info_score', 'clustering_accuracy']:
                    clustering_algorithm = clustering_algorithms[algorithm]
                    parameters_ranges = parameters_to_optimize[algorithm]

                    if algorithm == 'optimized_k_means':
                        results_test, results_train, best_experiment_config = fine_tune_k_means(
                            dev_dataset, test_dataset, #train_dataset
                            dev_features, test_features,#train_features
                            clustering_algorithm, parameters_ranges)
                    else:
                        results_test, results_train, best_experiment_config = fine_tune_unsupervised_clustering_parameters(
                            dev_dataset, test_dataset, #train_dataset
                            dev_features, test_features,#train_features
                            clustering_algorithm, parameters_ranges, optimization_criterion)

                    experiment_test = [{
                        'dataset': dataset,
                        'language_model': language_model_path,
                        'algorithm': algorithm,  #algorithm
                        'best_results': results_test,
                        'best_config': best_experiment_config,
                        'optimization_criterion': optimization_criterion
                    }]

                    experiment_train = [{
                        'dataset': dataset,
                        'language_model': language_model_path,
                        'algorithm': algorithm,
                        'best_results': results_train,
                        'best_config': best_experiment_config,
                        'optimization_criterion': optimization_criterion
                    }]

                    from_list_of_dict_to_jsonl(
                        f'experiment_results/{experiment_category}/experiments_unsupervised_clustering_open_baseline_datasets_test', 
                        experiment_test, write_on_existing_file=True)

                    from_list_of_dict_to_jsonl(
                        f'experiment_results/{experiment_category}/experiments_unsupervised_clustering_open_baseline_datasets_train', 
                        experiment_train, write_on_existing_file=True)

                        