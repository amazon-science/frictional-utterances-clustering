# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from frictional_utterances_clustering.utils.clustering_algorithms import *
from frictional_utterances_clustering.utils.clustering_metrics import *
from frictional_utterances_clustering.dataset_handling.dataset_utils import *
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import faiss

def launch_clustering_experiments(
    utterances_dataset, dict_of_unsupervised_clustering_algorithms,
    embeddings_column: str = 'utterance_embedding'):
    
    features = utterances_dataset[embeddings_column].to_list()
    feature_vectors = np.array(features)
    normalized_vectors = preprocessing.normalize(feature_vectors, norm="l2")

    results = {}
    
    for algorithm in dict_of_unsupervised_clustering_algorithms.keys():
        clustering_algorithm = dict_of_unsupervised_clustering_algorithms[algorithm]
        new_clusters = clustering_algorithm(normalized_vectors)
        utterances_dataset[f'utterance_intent_pred_{algorithm}'] = new_clusters
        
        gold_clusters, reconstructed_clusters = get_gold_and_predicted_clusters(
            utterances_dataset, name_intent_pred_column=f'utterance_intent_pred_{algorithm}')
        
        clustering_evaluation = compute_evaluation_metrics(gold_clusters, reconstructed_clusters)

        results[f"results_{algorithm}"] = clustering_evaluation

    return utterances_dataset, results

def evaluate_new_clusters(
    utterances_dataset: pd.DataFrame, new_clusters):

    utterances_dataset_eval = utterances_dataset.copy()
    
    utterances_dataset_eval.loc[:, 'utterance_intent_pred'] = new_clusters

    pred_cluster_assignments = utterances_dataset_eval['utterance_intent_pred'].to_list()
    gold_cluster_assignments = utterances_dataset_eval['utterance_intent'].to_list()

    gold_clusters, reconstructed_clusters = get_gold_and_predicted_clusters(utterances_dataset_eval)

    results = compute_evaluation_metrics(
        pred_cluster_assignments, gold_cluster_assignments,
        gold_clusters, reconstructed_clusters)

    return results

def upper_tri_indexing(A):
    n = A.shape[0]
    r,c = np.triu_indices(n,1)
    return A[r,c]

def tSNE(
    sentence_embeddings: np.ndarray, 
    intents: pd.core.series.Series
    ):
    pca = PCA(n_components=25)
    pca_result = pca.fit_transform(sentence_embeddings)
    tsne = TSNE(
        n_components=2, verbose=1, perplexity=50, n_iter=2000, early_exaggeration=50)
    tsne_results = tsne.fit_transform(pca_result)
    plot_tSNE_results = plt.figure(figsize=(32,18))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=intents,
        palette=sns.color_palette(n_colors=len(intents.unique())),
        legend="full"
    )
    
    plt.legend( 
        loc='upper left', 
        bbox_to_anchor=(0, -1, 1, 1),
        mode='expand',
        ncol = 4)
    
    return plot_tSNE_results

def get_average_similarity_per_intent_and_tSNE_plot(
    intent_dataset: pd.DataFrame, train_intents: List, dev_intents: List, 
    test_intents: List, frac_of_sample_to_use: float = 1,
    language_model_path: str='base_language_models/paraphrase-multilingual-mpnet-base-v2') -> Dict:
    
    test_dataset = intent_dataset[intent_dataset['utterance_split'] == 'test']
    
    test_sentence_embeddings = prepare_features_for_clustering(
        test_dataset, language_model=language_model_path)

    tSNE_plot = tSNE(test_sentence_embeddings, test_dataset["utterance_intent"])
    
    sentence_embeddings = prepare_features_for_clustering(
        intent_dataset, language_model=language_model_path)
    
    intent_dataset["utterance_embedding"] = list(sentence_embeddings)
    
    embeddings_by_intent = intent_dataset.groupby("utterance_intent")["utterance_embedding"]

    avg_similarity_by_intent = {
        'per_intent_statistics': {},
        'overall_statistics': {}
    }

    for intent_name, group in embeddings_by_intent:
        print(f"INTENT_NAME: {intent_name}")
        sample_embeddings = group.sample(
            frac=frac_of_sample_to_use, random_state=1).to_list()
        if len(sample_embeddings) == 0:
            sample_embeddings = [[0]*100]
        similarities = cosine_similarity(sample_embeddings, dense_output=False)
        upper_sim_values = upper_tri_indexing(similarities)
        if len(upper_sim_values) > 0:
            avg_similarity_by_intent[
                'per_intent_statistics'][intent_name] = float(upper_sim_values.mean())
        else:
            avg_similarity_by_intent[
                'per_intent_statistics'][intent_name] = "singleton"

    print('OVERALL STATISTICS')

    avg_similarity_by_intent['overall_statistics'][
        'total_average_infra_intent_pairwise_similarity'] = infra_intents_avg_cosine_similarity(intent_dataset)
    accuracy, avg_sim_at_one = precision_at_1_closer_retrieval(intent_dataset)
    avg_similarity_by_intent['overall_statistics']['total_retrieval_accuracy_at_1'] = accuracy
    avg_similarity_by_intent['overall_statistics']['total_retrieval_similarity_at_1'] = accuracy
    
    for split, intents in zip(['train', 'dev', 'test'], [train_intents, dev_intents, test_intents]):
        dataset = intent_dataset[intent_dataset['utterance_intent'].isin(intents)]
        avg_similarity_by_intent['overall_statistics'][
            f'{split}_average_infra_intent_pairwise_similarity'] = infra_intents_avg_cosine_similarity(dataset)
        accuracy, avg_sim_at_one = precision_at_1_closer_retrieval(dataset)
        avg_similarity_by_intent['overall_statistics'][f'{split}_retrieval_accuracy_at_1'] = accuracy
        avg_similarity_by_intent['overall_statistics'][f'{split}_retrieval_similarity_at_1'] = avg_sim_at_one

    return avg_similarity_by_intent, tSNE_plot

def precision_at_1_closer_retrieval(intent_dataset: pd.DataFrame):
    utterance_embeddings = intent_dataset["utterance_embedding"].to_list()
    intents = intent_dataset["utterance_intent"].to_list()
    
    embeddings = np.array(utterance_embeddings).astype("float32")
    faiss.normalize_L2(x=embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(embeddings, k=2)
    
    average_sim = np.mean(D[:, 1])
    
    indices_of_closer = I[:, 1]
    
    successfully_retrieved = 0
    
    for i in range(len(intents)):
        retrieved_utterance = indices_of_closer[i]
        if intents[i] == intents[retrieved_utterance]:
            successfully_retrieved += 1
    
    accuracy = successfully_retrieved/len(intents)
    
    print(accuracy, average_sim)
    
    return float(accuracy), float(average_sim)
    

def infra_intents_avg_cosine_similarity(intent_dataset: pd.DataFrame):
 
    utterance_embeddings = intent_dataset["utterance_embedding"].to_list()
    
    all_sim = cosine_similarity(utterance_embeddings, dense_output=False)
    
    intents_to_encode = intent_dataset["utterance_intent"].to_list()
    intents = encode_intents(intents_to_encode)
    
    pairwise_class_equality_negative = torch.ne(
            intents[None, :], intents[:, None]).float()
    
    all_sim = all_sim*np.array(pairwise_class_equality_negative)

    return float(np.sum(all_sim)/torch.sum(pairwise_class_equality_negative))

def encode_intents(utterance_intent_list: List):
        intents = list(set(utterance_intent_list))
        intents.sort()
        intent_encoding = {}
        counter = 0          #  TODO: refactor with collections.Counter
        for intent in intents:
            intent_encoding[intent] = counter
            counter += 1
        
        encoded_intents = []
        
        for intent in utterance_intent_list:
            encoded_intents.append(intent_encoding[intent])
        
        encoded_intents = torch.FloatTensor(encoded_intents)
        
        return encoded_intents

def prepare_features_for_clustering(
    utterances_dataframe: pd.DataFrame, 
    language_model: str = 'base_language_models/paraphrase-multilingual-mpnet-base-v2',
    name_utterances_column: str = 'utterance_text'):
    
    utterances = utterances_dataframe[name_utterances_column].to_list()
    features = get_sentence_embeddings(utterances, language_model)
    feature_vectors = np.array(features)
    normalized_vectors = preprocessing.normalize(feature_vectors, norm="l2")

    return normalized_vectors


def fine_tune_unsupervised_clustering_parameters(
    train_dataset, test_dataset, #train_dataset, 
    train_features, test_features, #train_features,
    algorithm, algorithm_param_ranges_to_optimize,
    optimization_criterion):

    experiment_list = list(product_dict(**algorithm_param_ranges_to_optimize))

    results = {}
    dict_to_compare_experiments = {}
    
    for count, experiment_hyperparameters in enumerate(experiment_list):
        
        print("EXPERIMENT NUMBER: ", count/len(experiment_list))

        new_clusters = algorithm(train_features, **experiment_hyperparameters)
        metrics_dict = evaluate_new_clusters(train_dataset, new_clusters)
        results[count] = metrics_dict
        dict_to_compare_experiments[count] = metrics_dict[optimization_criterion]
    
    best_experiment = max(dict_to_compare_experiments, key=dict_to_compare_experiments.get)

    best_experiment_config = experiment_list[best_experiment]

    test_clusters = algorithm(test_features, **best_experiment_config)
    final_metrics_dict_on_test = evaluate_new_clusters(test_dataset, test_clusters)

    train_clusters = algorithm(train_features, **best_experiment_config)
    final_metrics_dict_on_train = evaluate_new_clusters(train_dataset, train_clusters)

    return final_metrics_dict_on_test, final_metrics_dict_on_train, best_experiment_config

def fine_tune_k_means(
    train_dataset, test_dataset, #train_dataset, 
    train_features, test_features, #train_features,
    algorithm, algorithm_param_ranges_to_optimize):
    
    experiment_list = list(product_dict(**algorithm_param_ranges_to_optimize))

    best_experiment_config = experiment_list[0]

    test_clusters = algorithm(test_features, **best_experiment_config)
    final_metrics_dict_on_test = evaluate_new_clusters(test_dataset, test_clusters)

    train_clusters = algorithm(train_features, **best_experiment_config)
    final_metrics_dict_on_train = evaluate_new_clusters(train_dataset, train_clusters)

    return final_metrics_dict_on_test, final_metrics_dict_on_train, best_experiment_config
    

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

mean_var = lambda x: str(
    np.mean(x).round(decimals=2))+u"\u00B1"+str(np.std(x).round(decimals=2))

concatenate_strigs = lambda x: '|'.join(str(v) for v in x)


if __name__ == '__main__':
    print('implement unit testing')

