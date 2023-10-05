# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from frictional_utterances_clustering.dataset_handling.dataset_utils import *
from frictional_utterances_clustering.utils.experiments_utils import *

for split in ['test', 'train']:

    experiment_category = 'monolingual'
    
    experimental_results = import_jsonl_as_list_of_dict(
        f'experiment_results/{experiment_category}/experiments_unsupervised_clustering_open_baseline_datasets_{split}')

    experiments = []

    def post_process_experiment_result(experiment: Dict):
        experiment.update(experiment["best_results"])
        experiment.update(experiment["best_config"])
        del experiment["best_results"]
        del experiment["best_config"]
        return experiment

    for experiment in experimental_results:
        processed_results = post_process_experiment_result(experiment)
        experiments.append(processed_results)

    df = pd.DataFrame(experiments)

    aggregation_function = np.mean #np.mean #mean_var
    
    table = pd.pivot_table(
        df, values=['purity', 'recall', 'f1', 'adjusted_rand_index','adjusted_mutual_info_score', 
        'homogeneity', 'completeness','v_measure', 'clustering_accuracy', 'num_pred_clusters', 'num_gold_clusters'], 
        index=['dataset', 'algorithm', 'language_model', 'optimization_criterion'], 
        aggfunc={
            'purity': aggregation_function, 
            'recall': aggregation_function, 
            'f1': aggregation_function, 
            'adjusted_rand_index': aggregation_function,
            'adjusted_mutual_info_score': aggregation_function, 
            'homogeneity': aggregation_function, 
            'completeness': aggregation_function,
            'v_measure': aggregation_function,
            'clustering_accuracy': aggregation_function,
            'num_pred_clusters': aggregation_function,
            'num_gold_clusters': aggregation_function,
        })

    table_hyperparameters = pd.pivot_table(
        df, values=['eps', 'min_samples', 'linkage', #'max_clusters', 'interval_step', 
                    'distance_threshold', 'cut_threshold'], index=[
            'algorithm', 'dataset', 'language_model'], 
        aggfunc={
            'linkage': concatenate_strigs,
            'distance_threshold': concatenate_strigs,
            'eps': concatenate_strigs,
            'min_samples': concatenate_strigs,
            #'max_clusters': concatenate_strigs,
            #'interval_step': concatenate_strigs,
            'cut_threshold': concatenate_strigs
        })

    table.to_csv(
        f"experiment_results/{experiment_category}/experimental_results_{split}.csv", decimal=',', float_format='%.3f')
    table_hyperparameters.to_csv(
        f"experiment_results/{experiment_category}/experimental_hyperparameters_{split}.csv", decimal=',', float_format='%.3f')
        