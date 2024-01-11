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

## How to install on an AWS EC2 instance
1. On your laptop, git clone 
   - `git clone git@github.com:amazon-science/frictional-utterances-clustering.git`
2. Switch to cli_production_branch
   - `git checkout cli_production_branch`
3. copy the Frictional_Utterances_Clustering repository on the remote AWS EC2 instance
   - `scp -r frictional_utterances_clustering p3instance-useast:/home/ubuntu/`
4. ssh on the remote instance (e.g. p3instance-useast)
   - `ssh p3instance-useast`
5. cd into the project folder
   - `cd ~frictional_utterances_clustering`
6. install the package named frictional_utterances_clustering needed to run the unsupervised clustering experiments.
   - `python setup.py install`
7. Install the needed required libraries (if you haven’t already):
   - `pip install -r requirements.txt`
8. Download some base sentence model encoders from Hugging Face
   - `python downlad_base_sentence_encoders.py`

   The downloaded sentence encoders will be saved in the folder `frictional_utterances_clustering/base_language_models`
9. Copy the language model you want to use from the folder `base_language_models` into the folder folder `fine_tuned_language_models`
   - `cp -r base_language_models/bert-base-multilingual-cased fine_tuned_language_models/`
9. Run the following command to launch the unsupervised clustering experiments:
   - `PYTHONPATH=. python3 ./src/frictional_utterances_clustering/experiments_main.py`
10. The results will be stored in two dirs:
    1. the folder above will contain the results on the validation set:
       - `experiment_results/experiments_unsupervised_clustering_open_baseline_datasets_train`
    2. the folder above will contain the results on the test set:
       - `experiment_results/experiments_unsupervised_clustering_open_baseline_datasets_test`

<!--
## How to use it
To generate the datasets:
- run the scripts inside src/dataset_handling for all included  datasets
- run src/frictional_utterances_clustering/dataset_handling/datasets_basic_statistics.py
- run src/frictional_utterances_clustering/prepare_data_for_constrastive_supervised_clustering_learning.py
- run src/frictional_utterances_clustering/dataset_handling/inter_intra_intent_avg_similarity_calculation.py
- run src/frictional_utterances_clustering/datasets_statistical_analysis.py
- run src/frictional_utterances_clustering/language_models_evaluator.py
- run src/frictional_utterances_clustering/experiment_main.py
-->
## Walthrough of the Code for the Clusterign Experiments

The script `experiments_main.py` contains the code for running the clustering experiments. During a clustering experiment, the program:
1. [Loads the utternces in the validation and test set of the selected dataset or datasets](#10-loading-the-validation-and-test-set);
2. [Transform these utterances in their embeddings representations](#20-deriving-the-embeddings-representations-of-utterances);
3. [Repeatedly perform the grouping of utterances in the validation set in order to find the best hyper-parameters maximing the clustering clustering on the validation set](#30-performing-hyperparameter-selection-on-the-validation-set);
4. [Perform the final clustering of test; utterances using the optimal hyper-paramters found at step 3](#40-perform-the-final-clustering-on-the-test-set-using-the-optimal-hyper-parameters);
5. [Returns the clutering accuracy of the final clustering on the test set](#50-returns-the-clutering-accuracy-of-the-final-clustering-on-the-test-set)

##### 1.0 Loading the validation and test set

When the experiments start, the name of the datasets to use will be read from the `datasets` variable.
```python
datasets = ['Massive', ]
```

For each selected dataset, the  data split containg the validation set is read from file `dev.csv`  and stored in the variable `dev_dataset`. The latter is used to to select the hyper-parameters that results in the best clustering on the validation set.
```python
dev_dataset = pd.read_csv(f"data/Final_Datasets_For_Experiments/{dataset}/dev.csv")
```

Similarly, the data split containing the test set is read from file  `test.csv` and stored in the variable `test_dataset`:
```python
test_dataset = pd.read_csv(f"data/Final_Datasets_For_Experiments/{dataset}/test.csv")
```

This will be used to perform the clustering of utterances in the test and measure the final accuracy of the clustering algorithm.

**Data Sampling**. The parameter `fract_data_to_use` in the body of the script specifies the percentage of utterances that will be used in  the experiments. If the `fract_data_to_use` value is smaller than `1.0`, then the dataset is  downsampled to match the specified fraction value
This paramer is expecially useful for reducing the time needed to perform experiments on large test sets.
 

#####  2.0 Deriving the embeddings representations of utterances

The process of extracting the embeddings corresponding to utterances in the validation and test set is performed by the function `prepare_features_for_clustering`. 

```python
dev_features = prepare_features_for_clustering(
   dev_dataset, language_model=language_model_path)
test_features = prepare_features_for_clustering(
   test_dataset, language_model=language_model_path)
```

The function `prepare_features_for_clustering` takes in input a dataframe containing a list of the utterances and the name of the language model to use for extracting the embedings corresponding to the input utterances.
Then, it returns an *L<sub>2</sub>*-normlized version of the embeddings corresponding to the input utterances, which will be stored in the objects  `dev_features` and `test_features` for the validaton and test set, respectively.

```python
def prepare_features_for_clustering(
    utterances_dataframe: pd.DataFrame, 
    language_model: str = 'base_language_models/paraphrase-multilingual-mpnet-base-v2',
    name_utterances_column: str = 'utterance_text'):
    
    utterances = utterances_dataframe[name_utterances_column].to_list()
    features = get_sentence_embeddings(utterances, language_model)
    feature_vectors = np.array(features)
    normalized_vectors = preprocessing.normalize(feature_vectors, norm="l2")

    return normalized_vectors
```

Note that during the embeddings extraction process, the program will output the message `EXTRACTING THE FEATURES`.
 

##### 3.0 Performing hyperparameter selection on the validation set

The variable `clustering_algorithms` containing the list of algorithms that will be run in the experiments.

```python
clustering_algorithms = {
   'connected_componentes': connected_components,
   'DBSCAN': DBSCAN,
}
```

Each clustering algorithm is assoiated with a list of hyperparameters, whose range of possible values is defind in the dict object `parameters_to_optimize`

```python
parameters_to_optimize = {
    'connected_componentes':  {
        'cut_threshold': [
            0.3, 0.50, 1.0],
    },
    'DBSCAN': {
        'eps': [
            0.05, 0.50, 1], 
        'min_samples': [2, 5, 10, 15, 20, 25, 30],
    },
}
```

In order to find the hyperparameters that maximize the accuracy of the predicted clustering (measured on the validation set), we run the function `fine_tune_unsupervised_clustering_parameters` on the list of algorithms (`clustering_algorithms`) and the associated hyperparamters (`parameters_to_optimize`) defined above.

```python
for algorithm in clustering_algorithms.keys():
   for optimization_criterion in ['adjusted_mutual_info_score', 'clustering_accuracy']:
      clustering_algorithm = clustering_algorithms[algorithm]
      parameters_ranges = parameters_to_optimize[algorithm]

      results_test, results_train, best_experiment_config = fine_tune_unsupervised_clustering_parameters(
         dev_dataset, test_dataset, 
         dev_features, test_features,
         clustering_algorithm, parameters_ranges, optimization_criterion
      )
```

To maximize the accuracy of the predicted clustering, we must provide in input to the function also the measure we want to optimize, such as **Clustering Accuracy** or **Adjusted Mutual Information Score**.

When the hyperparameter optimization process starts, the program will output the message: `EXPERIMENT NUMBER:`

**Hyperparameters search**.  During the hyperparameter optimization process, the function `fine_tune_unsupervised_clustering_parameters` will repeatedly perform clustering according to the algorithm selected and the specific set of hyper-parametrs values under consideration at each optimization step. 

```python
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
```

The set of predicted clusters corresponding the specific algorithm-hyperparameters pair being examined is stored in the variable `new_clusters`, while the obejct `dict_to_compare_experiments` keep store the metric values results for each pair.
When the function execution ends, only the configuration of the experiment with the best result is selected, which is the one giving the best results on the validation set according to the selected optimization metrics. Then, the function use the optimal validation hyperparameters to compute and return the final results on the test set. 
These best hyperparameters are stored in the object `best_experiment_config`. Instead, the final results on validation and test sets are stored in the objects `final_metrics_dict_on_train` and `final_metrics_dict_on_test`, respectively.


##### 4.0 Performing the final clustering on the test set using the optimal hyper-parameters

The function `fine_tune_unsupervised_clustering_parameters` will return the best clustering results on the validation set and test set, which will be stored repsectively in the `results_train` and `results_test` objects in the main program.

```python
results_test, results_train, best_experiment_config = fine_tune_unsupervised_clustering_parameters(
   dev_dataset, test_dataset, #train_dataset
   dev_features, test_features,#train_features
   clustering_algorithm, parameters_ranges, optimization_criterion
)
```

Then, the best results onthe validation are saved to file:
```python
experiments_unsupervised_clustering_open_baseline_datasets_train.
```

Similarly, the final results on the test set are saved to file:
```python
experiments_unsupervised_clustering_open_baseline_datasets_test
```

##### 5.0 Returning the clutering accuracy of the final clustering on the test set

The function `evaluate_new_clusters` takes in input the  dataset containing utterances to cluster (`utterances_dataset`) and the clustering predicted by the model (`pred_clusters`) and returns an object (`results`)  containing a set of metrics values measuring the predicted clustering accuracy.

To do this, the function `evaluate_new_clusters` internally computes the assignment of utterances both to the gold clusters (`gold_cluster_assignments`) and predicted clusters (`pred_cluster_assignments`). Each utterance id is associated to a cluster id, which is ia number identifying the cluster it belongs to (e.g. `utterance001` &rarr; `cid01` )

These cluster assignments are then passed to the function `get_gold_and_predicted_clusters` to reconstruct the reference and predicted clusters, which are stored in the `gold_clusters` and `reconstructed_clusters` objects, respectively.

Finally, the `gold_clusters` and `reconstructed_clusters` objects are passed to the `compute_evaluation_metrics`, which computes micro and macro-average version of the provided clustering evaluation metrics, such as Clustering Accuracy, AMIS, etc..

```python
def evaluate_new_clusters(utterances_dataset: pd.DataFrame, pred_clusters):
   utterances_dataset_eval = utterances_dataset.copy()
   utterances_dataset_eval.loc[:, 'utterance_intent_pred'] = pred_clusters

   pred_cluster_assignments = utterances_dataset_eval['utterance_intent_pred'].to_list()
   gold_cluster_assignments = utterances_dataset_eval['utterance_intent'].to_list()

   gold_clusters, reconstructed_clusters = get_gold_and_predicted_clusters(utterances_dataset_eval)

   results = compute_evaluation_metrics(
        pred_cluster_assignments, gold_cluster_assignments,
        gold_clusters, reconstructed_clusters)

   return results
```

**Evaluation Metrics**. The  metrics results returned by the function `compute_evaluation_metrics` function are stored in a dict object. The dict keys are the name of clustering evaluation metrics such as Clustering Precision, Recall, F<sub>1</sub> score, Clustering Accuracy and Adjusted Mutual Information Score. The dict values instead correpond to the metric values.

Please, notice that the clustering version of the Precision, Recall and F<sub>1</sub> metrics is different from the corresponding classification version of the metrics. A complete definition of the clustering version of precision, recall, f<sub>1</sub> metrics can be found in [[1]](#1).


#####References
<a id="1">[1]</a> Iryna Haponchyk, Antonio Uva, Seunghak Yu, Olga Uryupina, and Alessandro Moschitti. 2018. [Supervised Clustering of Questions into Intents for Dialog System Applications](https://aclanthology.org/D18-1254/). In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2310–2321, Brussels, Belgium. Association for Computational Linguistics.
