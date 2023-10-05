# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import FrozenSet, List, Union, Dict, List, Tuple, Any, Counter as CounterT
import numpy as np
from numpy import mean
from sklearn import metrics
from collections import Counter, defaultdict
from scipy.optimize import linear_sum_assignment

def compute_evaluation_metrics(
    pred_cluster_assignments: List[int], gold_cluster_assignments: List[int],
    clusters_pred: List[FrozenSet], clusters_gold: List[FrozenSet], average=None):
    if average is None: 
        purity = compute_cluster_purity(clusters_pred, clusters_gold)
        recall = compute_cluster_recall(clusters_pred, clusters_gold)
        f1 = compute_cluster_f1(clusters_pred, clusters_gold)

    elif average== 'micro':
        assert isinstance(clusters_pred, list)
        assert isinstance(clusters_gold, list)

        purity, recall, f1 = micro_average(clusters_pred, clusters_gold)

    elif average=='macro':
        assert isinstance(clusters_pred, list)
        assert isinstance(clusters_gold, list)

        purity, recall, f1 = macro_average(clusters_pred, clusters_gold)

    else:
       raise IOError()

    adjusted_rand_index = metrics.adjusted_rand_score(
        gold_cluster_assignments, pred_cluster_assignments)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(
        gold_cluster_assignments, pred_cluster_assignments)

    homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(
        gold_cluster_assignments, pred_cluster_assignments)

    accuracy_calculator = ClusteringAccuracy()

    clustering_accuracy = accuracy_calculator.compute_metric(
        cluster_labels=pred_cluster_assignments, 
        reference_labels=gold_cluster_assignments
    )

    num_pred_clusters = len(set(pred_cluster_assignments))
    num_gold_clusters = len(set(gold_cluster_assignments))
    
    metrics_dict = {
        'purity': purity,
        'recall': recall,
        'f1': f1,
        'adjusted_rand_index': adjusted_rand_index,
        'adjusted_mutual_info_score': adjusted_mutual_info_score,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'clustering_accuracy': clustering_accuracy,
        'num_pred_clusters': num_pred_clusters,
        'num_gold_clusters': num_gold_clusters,
    }

    return metrics_dict

def compute_cluster_purity(
    clusters_pred: List[FrozenSet], clusters_gold: List[FrozenSet]):
    '''Compute cluster purity.
       >>> clusters_gold = [frozenset({'1', '3', '2'})]
       >>> clusters_pred = [frozenset({'1', '3'}), frozenset({'2'})]
       >>> compute_cluster_purity(clusters_pred, clusters_gold)
       1.0
    '''
    elements_num = 0
    purities = []
    for cluster_pred in clusters_pred:
        elements_num += len(cluster_pred)
        pi = max([len(cluster_pred & cluster_gold) for cluster_gold in clusters_gold] + [0])
        purities.append(pi)
    if sum(purities) <= 0: 
        return 0
    purity = 1.0 * sum(purities) / elements_num
    return purity

def compute_cluster_recall(
    clusters_pred: List[FrozenSet], clusters_gold: List[FrozenSet]):
    '''Compute cluster recall.
       >>> clusters_gold = {frozenset({'1', '3', '2'})}
       >>> clusters_pred = {frozenset({'1', '3'}), frozenset({'2'})}
       >>> recall = compute_cluster_recall(clusters_pred, clusters_gold)
       >>> recall - 0.66666666666667 < 1e-9
       True
    ''' 
    elements_num = 0
    recalls = []
    for cluster_gold in clusters_gold:
        elements_num += len(cluster_gold)
        ri = max([len(cluster_gold & cluster_pred) for cluster_pred in clusters_pred] + [0])
        recalls.append(ri)
    if sum(recalls) <= 0:
        return 0
    recall = 1.0 * sum(recalls) / elements_num
    return recall

def compute_cluster_f1(
    clusters_pred: List[FrozenSet], clusters_gold: List[FrozenSet]):
    '''Compute cluster f1.
       >>> clusters_gold = {frozenset({'1', '3', '2'})}
       >>> clusters_pred = {frozenset({'1', '3'}), frozenset({'2'})}
       >>> compute_cluster_f1(clusters_pred, clusters_gold)
       0.8
    '''
    purity = compute_cluster_purity(clusters_pred, clusters_gold)
    recall = compute_cluster_recall(clusters_pred, clusters_gold)
    if purity + recall == 0:
        return 0
    else:
        return 2 * purity * recall / (purity + recall)
        
def micro_average(clusters_pred_list, clusters_gold_list):
    '''Perform micro-average on a list of evaluation metrics.'''
    total_gold_clusters = set.union(*clusters_gold_list)
    total_pred_clusters = set.union(*clusters_pred_list)
    purity, recall, f1 = compute_evaluation_metrics(total_pred_clusters, total_gold_clusters)
    return purity, recall, f1

def macro_average(clusters_pred_list, clusters_gold_list):
    '''Perfrom macro-average on a list of evalustion metrics.'''
    purities = []
    recalls = []
    f1s = []
    for clusters_pred, clusters_gold in zip(clusters_pred_list, clusters_gold_list):
        purity, recall, f1 = compute_evaluation_metrics(clusters_pred, clusters_gold)
        purities.append(purity)
        recalls.append(recall)
        f1s.append(f1)
    return mean(purities), mean(recalls), mean(f1s)


def compute_optimal_alignment(predicted_labels: List[str], reference_labels: List[str]) -> Dict[str, str]:
    """
    Find an optimal assignment of predicted labels (e.g. cluster labels) to corresponding reference
    labels (ground truth labels) by maximizing overlap between each predicted and ground truth label.
    :param predicted_labels: predicted labels, e.g. cluster IDs
    :param reference_labels: corresponding reference labels, such as ground truth cluster labels
    :return: mapping of predicted labels to reference labels
    """
    # (1) assign unique labels to indices
    unique_predicted_labels, cluster_label_indices = np.unique(predicted_labels, return_inverse=True)
    unique_ref_labels, reference_label_indices = np.unique(reference_labels, return_inverse=True)

    # (2) build matrix counting overlap between predicted and reference labels
    cost_matrix = np.zeros((len(unique_predicted_labels), len(unique_ref_labels)))
    for predicted, reference in zip(cluster_label_indices, reference_label_indices):
        cost_matrix[predicted][reference] += 1

    # (3) compute optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    # (4) return optimal 1:1 mapping of cluster labels to reference labels
    return {
        unique_predicted_labels[row]: unique_ref_labels[col] for row, col in zip(row_ind.tolist(), col_ind.tolist())
    }


def align_labels(predicted_labels: List[str], alignment: Dict[str, str], default_label=None) -> List[str]:
    """
    Apply alignment to predicted labels.
    :param predicted_labels: predicted labels, e.g. cluster IDs
    :param alignment: alignment of predicted labels to reference labels
    :param default_label: default label to be used if predicted label is not present in alignment
    :return: aligned predicted labels
    """
    return [alignment.get(label, default_label) for label in predicted_labels]


def count_cluster_label_overlap(
    first_clustering: List[str], second_clustering: List[str]
) -> Dict[str, CounterT[str]]:
    """
    Return the label overlap counts between two clusterings.
    """
    overlap_counts = defaultdict(Counter)
    for first_label, second_label in zip(first_clustering, second_clustering):
        overlap_counts[first_label][second_label] += 1
    return overlap_counts


class ClusteringMetric(object):

    def metric_name(self) -> str:
        """
        Returns the name of the clustering metric for reporting.
        """
        raise NotImplementedError

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        """
        Compute extrinsic cluster metric given cluster labels and corresponding reference (ground truth) labels.
        :param cluster_labels: predicted cluster labels
        :param reference_labels: ground truth labels
        :return: cluster metric result
        """
        raise NotImplementedError
    

class ClusteringAccuracy(ClusteringMetric):
    """
    Clustering accuracy, in which an optimal 1:1 alignment is found between predicted cluster labels
    and reference labels.
    """

    def metric_name(self) -> str:
        return 'ACC'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        alignment = compute_optimal_alignment(cluster_labels, reference_labels)
        aligned_labels = align_labels(cluster_labels, alignment)
        total_correct = sum(1 for aligned, reference in zip(aligned_labels,
                                                            reference_labels) if aligned == reference)
        accuracy = total_correct / len(reference_labels) if reference_labels else 0
        return accuracy