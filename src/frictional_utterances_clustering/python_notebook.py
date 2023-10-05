# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from frictional_utterances_clustering.utils.experiments_utils import *
from frictional_utterances_clustering.utils.clustering_metrics import *
from frictional_utterances_clustering.utils.clustering_algorithms import *
from frictional_utterances_clustering.dataset_handling.dataset_utils import *

datasets_languages_and_intents = load_dict_from_json(
    "data/Datasets_Analysis/datasets_intents_and_languages.json")

# datasets = ['Snips', 'pathicKnowledge', 'MultiAtis', 'Massive']

# for dataset in datasets:
#     train_intents, dev_intents, test_intents = split_intents_in_train_dev_test(
#         datasets_languages_and_intents[dataset]["intents"], [0.6, 0.2, 0.2])
    
#     datasets_languages_and_intents[dataset]["train_intents"] = list(train_intents)
#     datasets_languages_and_intents[dataset]["dev_intents"] = list(dev_intents)
#     datasets_languages_and_intents[dataset]["test_intents"] = list(test_intents)

#     print(len(train_intents), len(dev_intents), len(test_intents))

    
# save_dict_as_json(
#     datasets_languages_and_intents, "data/Datasets_Analysis/datasets_intents_and_languages_2.json")

import pandas as pd 

all_language = []

for dataset in ['Snips', 'pathicKnowledge', 'MultiAtis', 'Massive']:
    
    languages = datasets_languages_and_intents[dataset]["languages"]
    all_language.extend(languages)
    print(dataset)
    
    train = pd.read_csv(
        f'data/Final_Datasets_For_Experiments/{dataset}/train.csv', 
        usecols = [
            'utterance_intent', 'utterance_text', 
            'dataset', 'utterance_lang'])
    dev = pd.read_csv(
        f'data/Final_Datasets_For_Experiments/{dataset}/dev.csv',
        usecols = [
            'utterance_intent', 'utterance_text', 
            'dataset', 'utterance_lang'])
    test = pd.read_csv(
        f'data/Final_Datasets_For_Experiments/{dataset}/test.csv',
        usecols = [
            'utterance_intent', 'utterance_text', 
            'dataset', 'utterance_lang'])

    print('total dataset')

    print(len(train))
    print(len(dev))
    print(len(test))

    print('intents in total dataset')

    print(len(train['utterance_intent'].unique()))
    print(len(dev['utterance_intent'].unique()))
    print(len(test['utterance_intent'].unique()))

    # 35807
    # 34144
    # 40230
    # 893
    # 670
    # 671

    # massive_train = train[train['dataset']=='massive']
    # massive_dev = dev[dev['dataset']=='massive']
    # massive_test = test[test['dataset']=='massive']

    # print('massive dataset')

    # print(len(massive_train[massive_train['utterance_lang']=='US']))
    # print(len(massive_dev[massive_dev['utterance_lang']=='US']))
    # print(len(massive_test[massive_test['utterance_lang']=='US']))

    # print('intents in massive dataset')

    # print(len(massive_train[massive_train['utterance_lang']=='US']['utterance_intent'].unique()))
    # print(len(massive_dev[massive_dev['utterance_lang']=='US']['utterance_intent'].unique()))
    # print(len(massive_test[massive_test['utterance_lang']=='US']['utterance_intent'].unique()))

print(all_language)
print(list(set(all_language)))