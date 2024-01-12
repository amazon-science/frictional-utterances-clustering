#!/bin/bash
# Script: print_results.sh
# Description: Takes a results file in JSON format as input and prints it in human-readable TSV format.
# Usage: ./print_results.sh <json_file>

# Check if the user provided a JSON file path as an argument
if [ "$#" -ne 1 ]; then
  echo "Error: Please provide a JSON file path as a command-line argument."
  echo "Usage: $0 <json_file>"
  exit 1
fi

json_file=$1

jq -r '"\(.algorithm)\t\(.language_model)\t\(.best_results.clustering_accuracy)\t\(.best_results.purity)\t\(.best_results.recall)\t\(.best_results.f1)\t\(.best_config.distance_threshold)\t\(.best_results.num_pred_clusters)\t\(.best_results.num_gold_clusters)"' ${json_file} | sort -nrk 3 | awk 'BEGIN{print "algorithm\tlanguage_model\tclustering_accuracy\tpurity\trecall\tf1\tdistance_threshold\tnum_pred_clusters\tnum_gold_clusters"}{print $0}' 