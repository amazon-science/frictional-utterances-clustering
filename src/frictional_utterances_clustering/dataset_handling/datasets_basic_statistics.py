# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd
import functools

from frictional_utterances_clustering.dataset_handling.dataset_utils import *

experiment_number = 4

custom_pivot = functools.partial(pd.pivot_table, 
        values='utterance_id', 
        index=['utterance_intent'], 
        columns=['utterance_lang'], 
        aggfunc=pd.Series.nunique)

datasets_name = os.listdir('data/Processed_Datasets/')

datasets_intents_and_languages = {}

manually_assigned_train_dev_test_intents = {}

for dataset in datasets_name:

        datasets_intents_and_languages.setdefault(dataset, {})
        
        list_of_records = join_all_jsonl_records_in_single_list(f"data/Processed_Datasets/{dataset}")
        df = pd.DataFrame(list_of_records)
        # datasets_intents_and_languages[dataset]['languages'] = list(df['utterance_lang'].unique())
        # datasets_intents_and_languages[dataset]['intents'] = list(df['utterance_intent'].unique())

        datasets_intents_and_languages[dataset]['languages'] = df.groupby(
                'utterance_lang').count().to_dict(orient='dict')['dataset']
        datasets_intents_and_languages[dataset]['intents'] = df.groupby(
                'utterance_intent').count().to_dict(orient='dict')['dataset']

        df_pivot = custom_pivot(data=df)

        df_pivot.to_csv(f'data/Datasets_Analysis/{dataset}_lang_intent_statistics.csv', index=True)

        if dataset in manually_assigned_train_dev_test_intents:
                print("manually pre-assigned splits")
                datasets_intents_and_languages[dataset][
                        "train_intents"] = manually_assigned_train_dev_test_intents[dataset]["train_intents"]
                datasets_intents_and_languages[dataset][
                        "dev_intents"] = manually_assigned_train_dev_test_intents[dataset]["dev_intents"]
                datasets_intents_and_languages[dataset][
                        "test_intents"] = manually_assigned_train_dev_test_intents[dataset]["test_intents"]
        else:
                train_intents, dev_intents, test_intents = split_intents_in_train_dev_test(
                        datasets_intents_and_languages[dataset]["intents"], [0.6, 0.2, 0.2])
                
                datasets_intents_and_languages[dataset]["train_intents"] = list(train_intents)
                datasets_intents_and_languages[dataset]["dev_intents"] = list(dev_intents)
                datasets_intents_and_languages[dataset]["test_intents"] = list(test_intents)

save_dict_as_json(
        datasets_intents_and_languages,
        f'data/Datasets_Analysis/datasets_intents_and_languages_{experiment_number}.json'
)

