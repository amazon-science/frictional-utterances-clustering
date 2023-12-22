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

## How to use it
To generate the datasets:
- run the scripts inside src/dataset_handling for all included datasets
- run src/frictional_utterances_clustering/dataset_handling/datasets_basic_statistics.py
- run src/frictional_utterances_clustering/prepare_data_for_constrastive_supervised_clustering_learning.py
- run src/frictional_utterances_clustering/dataset_handling/inter_intra_intent_avg_similarity_calculation.py
- run src/frictional_utterances_clustering/datasets_statistical_analysis.py
- run src/frictional_utterances_clustering/language_models_evaluator.py
- run src/frictional_utterances_clustering/experiment_main.py

## Code Walkthrough

### Experiments

The script `experiments_main.py` contains the code for running the clustering experiments.

The name of the dataset to use is stored in the `dataset` variable.

The path of the file containing the validation set is:

```
data/Final_Datasets_For_Experiments/{dataset}/labeled_utterances_for_for_hyperparameter_optimization.csv
```

The validation set is read from file and stored as a data frame in the `dev_dataset` variable

The pat of the file containing the test set is:

```
data/Final_Datasets_For_Experiments/{dataset}/new_unlabeled_utterances_to_cluster.csv'
```

The test set is read from the file and stored as a data frame in the `test_dataset` variable.

#### Data Sampling

The variable `fract_data_ti_use` specifies the percentage of utterances to use for the experiments. If `the fract_data_to_use` value is < 1.0, then the dataset is  downsampled to match the fraction of data specified.

#### Extracting features 

The function `prepare_features_for_clustering` extract the embeddings corresponding to the utterances stored in the  dataframes of the validation and test set. During the embeddings extraction process, the program will output the message: `EXTRACTING THE FEATURES`

The variables containing the extracted embeddings are called `dev_features` and `test_features`, rispectively.

#### Applying clustering algorithms

When the script `experiments_main.py` is launched, all the clustering experiments defined at the begin of the file are executed. For each algorithm, all the corresponding hyperparamters together with their range of values are stored in the `paramter_ranges` variable. The function `fine_tune_unsupervised_clustering_parameters` takes care of finding the hyperparameters values maximizing the selected optmization criterion, such as **Accuracy** or **Adjusted Mutual Information Score**. When the hyperparameter optimization process start, the program will output the message: `EXPERIMENT NUMBER:`

#### Results 

The `fine_tune_unsupervised_clustering_parameters` will return the results on the validation set (`results_train`), the results on the test set (`results_test`) and the best hyperparameter configuration (`best_experiment_config`). The results coresponding to the best parameters configuration are stored in the variables `experiment_train` and `experiment_test` for the validation and test set, respectively. 
Finally, the best results are saved on two files: 

1. The best results on the validation set are saved in:

```
experiment_results/experiments_unsupervised_clustering_open_baseline_datasets_train
```

2. The best results on the test set are saved in:

```
experiment_results/experiments_unsupervised_clustering_open_baseline_datasets_test
```

### Hyperparamters Optimization

The script `experiment_utils.py` contains all the code for performing optmization of the clustering algorithms' hyperparamaters.

In particular, the function `fine_tune_unsupervised_clustering_parameters` - already introduced in the previous section — performs the search over the space of hyper-parameters values defined by the user.

#### Hyper-parameters Search

The hyper-parameters values passed to the function anre stored in the parameter `algorithm_param_ranges_to_optimize`, which contains all possible combinations of values in a grid search fashion.

During the hyperparameter optimization process, the function repeatedly perform clustering according to the algorithm selected and the specific set of hyper-parametrs values under consideration at the current step. The the set of predicted clusters corresponding to specific algorithm-hyperparameters pair is stored in the variable `new_clusters`. All the intermediate clustering results are also stored in the variable `dict_to_compare_experiments`, which keep track all the intermediate experiments results.
At the end, only the configuration of the experiment with the best result is selected, which is the one giving the best results according to the  selected optimization metric (Clustering Accuracy or Adjusted Mutual Information Score). Then, the function returns the configuration of the best experiment, with the corresponding metrics computed on the validation and test set. These are stored in the variables `best_experiment_config`, `final_metrics_dict_on_train` and `final_metrics_dict_on_test`, respectively.

#### Evaluation of predicted clustering

The function `evaluate_new_clusters` takes in input the validation set and the predictions of the model. It then compute the assignment of utterances to the gold clusters `gold_cluster_assignments` and the the assignments of utterances to predicted clusters `pred_cluster_assignments`.
These assignments are then pased to the function `get_gold_and_predicted_clusters` to reconstruct the reference and predicted clusters, which are stored in the `gold_clusters` and `reconstructed_clusters` variables.

#### Evaluation Metrics

The gold and reconstructed clusters are passed to the `compute_evaluation_metrics` function, which will return the clustering metrics results, which are stored as dict in the `metric_dict` variable. The metrics_dict reports the name of clustering evaluation metrics such as Clustering *Precision*, *Recall*, *F<sub>1</sub> score*, *Accuracy*, etc.. together with their corresponding values.

