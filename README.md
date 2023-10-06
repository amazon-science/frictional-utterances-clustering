# Frictional Utterances Clustering
This is a package to apply clustering algorithms to utterances, 
embedded with a fine-tuned version out of the [Supervised Intent Clustering package](https://github.com/amazon-science/supervised-intent-clustering).

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

SPDX-License-Identifier: [CC-BY-NC-4.0](./LICENSE)

## How to cite us
```
@Inproceedings{Barnabo2023,
 author = {Giorgio Barnabo and Antonio Uva and Sandro Pollastrini and Chiara Rubagotti and Davide Bernardi},
 title = {Supervised clustering loss for clustering-friendly sentence embeddings: An application to intent clustering},
 year = {2023},
 url = {https://www.amazon.science/publications/supervised-clustering-loss-for-clustering-friendly-sentence-embeddings-an-application-to-intent-clustering},
 booktitle = {IJCNLP-AACL 2023},
}
```

## How to use it
To generate the datasets:
- run the scripts inside src/dataset_handling for all included datasets
- run src/frictional_utterances_clustering/dataset_handling/datasets_basic_statistics.py
- run src/frictional_utterances_clustering/prepare_data_for_constrastive_supervised_clustering_learning.py
- run src/frictional_utterances_clustering/dataset_handling/inter_intra_intent_avg_similarity_calculation.py
- run src/frictional_utterances_clustering/datasets_statistical_analysis.py
- run src/frictional_utterances_clustering/language_models_evaluator.py
- run src/frictional_utterances_clustering/experiment_main.py
