# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
from sklearn import cluster
from typing import List, Union
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as compute_connected_components

def optimized_k_means(
    normalized_feature_vectors: np.array, 
    max_clusters: int = 50,
    interval_step: int = 2):

    kmeanModel = cluster.KMeans(1).fit(normalized_feature_vectors)
    
    best_silhouette_score = -1
    best_label_assignment_so_far = kmeanModel.labels_
        
    for n_clusters in range(interval_step, max_clusters+1, interval_step):
        print(n_clusters)
        if n_clusters >= len(normalized_feature_vectors):
            break
        kmeanModel = cluster.KMeans(
            n_clusters=n_clusters).fit(normalized_feature_vectors)

        current_silhouette_score = silhouette_score(
            normalized_feature_vectors, kmeanModel.labels_)
            
        if current_silhouette_score > best_silhouette_score:
            best_silhouette_score = current_silhouette_score
            best_label_assignment_so_far = kmeanModel.labels_

    return best_label_assignment_so_far


def connected_components(normalized_feature_vectors: np.array, cut_threshold: float = 0.5) -> List:
    
    graph = cosine_similarity(normalized_feature_vectors, dense_output=False)
    
    graph[graph < cut_threshold] = 0
    
    graph = csr_matrix(graph)
    
    n_components, labels = compute_connected_components(
        csgraph=graph, directed=False, return_labels=True)
    
    return labels


def k_means(normalized_feature_vectors: np.array, n_clusters: int = 20) -> List:

    """
        - Input: feature matrix, number of clusters
        - Output: cluster assignments
        - Usecase: general-purpose, even cluster size, flat geometry,
        not too many clusters, inductive
    """

    kmeanModel = cluster.KMeans(
        n_clusters=n_clusters).fit(normalized_feature_vectors)

    current_distortion = sum(
        np.min(cdist(normalized_feature_vectors, kmeanModel.cluster_centers_,
        'euclidean'), axis=1)) / normalized_feature_vectors.shape[0]
    
    current_inertia = kmeanModel.inertia_
    
    print("current_distortion", current_distortion)
    print("current_inertia", current_inertia)
    
    return kmeanModel.labels_

def agglomerative_hierarchical_clustering(
    normalized_feature_vectors: np.array, n_clusters: Union[int, None] = None, 
    linkage: str = 'ward', distance_threshold: float = 0.5):
    """
        - Input:
        --- feature_vectors: array of normalized features (n, k)
        --- either n_clusters or distance_threshold
        --- linkage: ward, complete, single, average
        - Output: cluster assignments
        - Usecase: many clusters, possibly connectivity constraints, 
        transductive
    """    

    agglomerative_clustering_dendrogram = cluster.AgglomerativeClustering(
        n_clusters=n_clusters, distance_threshold=distance_threshold, 
        linkage = linkage).fit(normalized_feature_vectors)
    
    print(f"Number of clusters found: {agglomerative_clustering_dendrogram.n_clusters_}")
    
    return agglomerative_clustering_dendrogram.labels_


def DBSCAN(
    normalized_feature_vectors: np.array, eps: float = 1.2, 
    min_samples: int = 2):
    """
        - Input:
        --- feature_vectors: array of normalized features (n, k)
        - Usecase: non-flat geometry, uneven cluster sizes, outlier removal,
                    transductive
    """    

    DBSCAN_clusters = cluster.DBSCAN(
        eps=eps, min_samples=min_samples).fit(normalized_feature_vectors)
    
    #print(f"Coreset found: {DBSCAN_clusters.core_sample_indices_}")
    
    return DBSCAN_clusters.labels_


# sktlearn clustering algorithms: https://scikit-learn.org/stable/modules/clustering.html

# K-Means, Ward hierarchical clustering, Agglomerative clustering, DBSCAN, OPTICS*, BIRCH*, Bisecting K-Means*, RCC*

# DBSCAN (Density_Based Clustering)
# Agglomerative clustering: linkage = complete
# aggclust = (AgglomerativeClustering(n_clusters=None, affinity='cosine', 
#                                     compute_full_tree=True, distance_threshold=0.1, linkage='complete')
#             .fit_predict(sentences[utts_len_le2]))
# Robust Continuous Clustering (RCC): https://github.com/yhenon/pyrcc


# k-means with Bayesian Information Criterion (BIC)
# ADVIN


# TO CHECK

# - https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
# - https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-clus[%E2%80%A6]eans-silhouette-analysis-py


