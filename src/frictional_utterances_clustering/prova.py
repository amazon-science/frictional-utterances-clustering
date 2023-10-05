# FrictionalUtterancesClustering
# This is a package to apply clustering algorithms to utterances, 
# embedded with a fine-tuned version of SupervisedIntentClustering package.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

list_of_sentences = [
        'che cosa accade il giorno che dono il sangue', 
        'ce lo dici se adesso sei attivato']

model = SentenceTransformer(
         'base_language_models/paraphrase-multilingual-mpnet-base-v2')

sentence_embeddings = model.encode(
        list_of_sentences, batch_size=64, show_progress_bar=True)

print(sentence_embeddings.shape)

# Step 1: Change data type
embeddings = np.array(sentence_embeddings).astype("float32")

print(np.sum(embeddings**2, axis=1))

faiss.normalize_L2(x=embeddings)

print(np.sum(embeddings**2, axis=1))

print(embeddings.shape)

# Step 2: Instantiate the index
index = faiss.IndexFlatIP(embeddings.shape[1])

# Step 3: Pass the index to IndexIDMap
#index = faiss.IndexIDMap(index)

# Step 4: Add vectors and their IDs
index.add(embeddings)

user_query = ["Che succede quando si dona il sangue?"]
user_query = model.encode(user_query)

print(user_query.shape)

queries = np.array(user_query).astype("float32")
faiss.normalize_L2(x=queries)

print(queries.shape)

D, I = index.search(embeddings, k=2)

print(I[:, 1])
print(D[:, 1])

print(np.mean(D[:, 1]))
#print(f'L2 distance: {D}\n\nMAG paper IDs: {I}')