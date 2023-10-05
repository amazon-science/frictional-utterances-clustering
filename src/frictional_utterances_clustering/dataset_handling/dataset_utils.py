# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set, Tuple
import pandas as pd
import json
import glob
import os
import random

def get_sentence_embeddings(list_of_sentences: List[str], 
        hf_model_name_or_path: str='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):

    model = SentenceTransformer(hf_model_name_or_path)
    sentence_embeddings = model.encode(
        list_of_sentences, batch_size=64, show_progress_bar=True)
    return sentence_embeddings
    
def from_list_of_dict_to_jsonl(
    saving_path: str, list_of_dicts: List,
    write_on_existing_file: bool = False):
    
    if write_on_existing_file:
        type_of_writing = 'a'
    else:
        type_of_writing = 'w'
    
    with open(saving_path, type_of_writing) as f:
        for item in list_of_dicts:
            f.write(json.dumps(item) + "\n")

def import_jsonl_as_list_of_dict(
    path_to_file: str, sample_frac: float = 1):
    
    with open(path_to_file, 'r') as json_file:
        json_list = list(json_file)
    
    list_of_records = []
    
    for json_str in json_list:
        if random.random() <= sample_frac:
            record = json.loads(json_str)
            list_of_records.append(record)

    return list_of_records


def join_all_jsonl_records_in_single_list(
    path_to_jsonl_files: str,
    languages_to_keep: List = [],
    percentage_dataset_to_include: float = 1):

    all_files = glob.glob(path_to_jsonl_files + "/*.jsonl")
    all_records = []
    for file_path in all_files:
        file_name = os.path.basename(os.path.normpath(file_path))
        if (len(languages_to_keep) > 0 
        and not any(x in file_name for x in languages_to_keep)): continue

        all_records += import_jsonl_as_list_of_dict(
            file_path, sample_frac=percentage_dataset_to_include)
    return all_records


def save_dict_as_json(data: Dict, saving_path: str):
    with open(saving_path, 'w') as fp:
        json.dump(
            data, fp, 
            sort_keys = True, 
            indent=4)

def load_dict_from_json(path_to_file: str):
    with open(path_to_file, 'r') as fp:
        data = json.load(fp)
    return data


def split_intents_in_train_dev_test(
    intents: List, splits: List[str]) -> Tuple[Set, Set, Set]:

    intents = {k: v for k, v in intents.items() if v > 1}
    
    train_size = int(len(intents)*splits[0])
    dev_size = int(len(intents)*splits[1])
    
    all_intents = set(intents)
    train =  set(random.sample(all_intents, train_size))
    dev_test = all_intents - train
    dev = set(random.sample(dev_test, dev_size))
    test = dev_test - dev

    return train, dev, test

def assign_split_to_utterances(
    dataframe, train_intents: Set,
    dev_intents: Set, test_intents: Set) -> None:
    dataframe.loc[dataframe["utterance_intent"].isin(train_intents), "utterance_split"] = "train"
    dataframe.loc[dataframe["utterance_intent"].isin(dev_intents), "utterance_split"] = "dev"
    dataframe.loc[dataframe["utterance_intent"].isin(test_intents), "utterance_split"] = "test"

def get_gold_and_predicted_clusters(
    dataframe, name_intent_pred_column: str = 'utterance_intent_pred',
    name_intent_column: str = 'utterance_intent'):
    gold_groupby = dataframe.groupby(name_intent_column)
    pred_groupby = dataframe.groupby(name_intent_pred_column)

    list_of_gold_clusters = list(gold_groupby.groups.values())
    list_of_pred_clusters = list(pred_groupby.groups.values())

    gold_clusters = [frozenset(cluster) for cluster in list_of_gold_clusters]
    reconstructed_clusters = [frozenset(cluster) for cluster in list_of_pred_clusters]

    return gold_clusters, reconstructed_clusters

