# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from datasets import load_dataset, ClassLabel, Value
from typing import List, Dict

snips_dataset = load_dataset("snips_built_in_intents")

def reverse_label_encoding(example):
    label_names = ['ComparePlaces', 'RequestRide', 'GetWeather', 
                    'SearchPlace', 'GetPlaceDetails', 'ShareCurrentLocation', 
                    'GetTrafficInformation', 'BookRestaurant', 'GetDirections', 'ShareETA']
    example['label_encoding'] = label_names[example['label']]
    return example

for split, dataset in snips_dataset.items():
    dataset = dataset.map(reverse_label_encoding)
    dataset.to_json(f"data/Snips/{split}.jsonl")