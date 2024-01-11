import os
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModelForMaskedLM

# Get the absolute path of the directory where the Python script is located
current_directory = os.path.dirname(os.path.abspath(__file__))

# The GitHub project directory is one level above the script directory 'src'
github_project_directory = os.path.dirname(os.path.dirname(current_directory))

# Download and save locally the Paraphrase Multilingual Mpnet model

def load_model(model_cls, model_name):
    try:
        model = model_cls(model_name)
    except (ValueError, EnvironmentError):
        model = model_cls.from_pretrained(model_name)
    return model

def save_model(model, tokenizer, model_path, ):
    try:
        model.save(model_path)
        tokenizer.save(model_path)
    except AttributeError:
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

def download_and_save_model(tokenizer_cls, model_cls, model_name, model_dir, model_ns=None):
    """
    Downloads a model from Hugging Face using the specified tokenizer and model classes,
    saves it locally
     at the specified directory, and prints a message with a checkmark
    emoji upon successful saving.

    Args:
    - tokenizer_cls (type): The tokenizer class (e.g., AutoTokenizer) to use for the model.
    - model_cls (type): The model class (e.g., SentenceTransformer) to use for the model.
    - model_name (str): The name of the model to download.
    - model_dir (str): The directory path to save the downloaded model.
    - model_namespace (str): The namespace of the model on Hugging Face.

    """
    model_path = os.path.join(model_dir, model_name)
    model_path = f"{github_project_directory}/base_language_models/{model_name}"
    print(f"Downloading the model '{model_name}' from Hugging Face... ", flush=True, end="")
    full_model_name = f"{model_ns}/{model_name}" if model_ns is not None else model_name
    tokenizer = tokenizer_cls.from_pretrained(full_model_name)
    model = load_model(model_cls, f'{model_name}')
    print("✅")
    print(f"Saving the model at '{model_path}'... ", flush=True, end="")
    save_model(model, tokenizer, model_path)
    print("✅", end="\n\n")

SAVED_MODEL_DIR = f"{github_project_directory}/base_language_models"

#download_and_save_model(AutoTokenizer, SentenceTransformer, "paraphrase-multilingual-mpnet-base-v2", SAVED_MODEL_DIR, "sentence-transformers")
#download_and_save_model(AutoTokenizer, SentenceTransformer, "paraphrase-mpnet-base-v2", SAVED_MODEL_DIR, "sentence-transformers")
#download_and_save_model(BertTokenizer, BertModel, "bert-base-multilingual-cased", SAVED_MODEL_DIR)
#download_and_save_model(AutoTokenizer, AutoModelForMaskedLM, "xlm-roberta-base", SAVED_MODEL_DIR)

modelPath = "/home/ubuntu/frictional-utterances-clustering/base_language_models/paraphrase-multilingual-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model.save(modelPath)
tokenizer.save_pretrained(modelPath)

modelPath = "/home/ubuntu/frictional-utterances-clustering/base_language_models/paraphrase-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
model = SentenceTransformer('paraphrase-mpnet-base-v2')
model.save(modelPath)
tokenizer.save_pretrained(modelPath)

modelPath = "/home/ubuntu/frictional-utterances-clustering/base_language_models/bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.save_pretrained(modelPath)
tokenizer.save_pretrained(modelPath)

modelPath = "/home/ubuntu/frictional-utterances-clustering/base_language_models/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
model.save_pretrained(modelPath)
tokenizer.save_pretrained(modelPath)
