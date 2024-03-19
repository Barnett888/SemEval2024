import transformer_baseline
import datasets
import transformers
import random
import pandas as pd
import numpy as np
import spacy
import os
import torch
from IPython.display import display, HTML
from transformer_baseline import get_data
from transformer_baseline import preprocess_function
from transformers import AutoTokenizer
from transformer_baseline import fine_tune
from transformer_baseline import test
from transformer_baseline import compute_metrics
from transformers import AutoModel
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForMaskedLM

os.chdir("C:/Users/94248/Desktop/SemEval2024_Task8/")
cwd = os.getcwd()

train_path = cwd+"/subtaskA_train_monolingual.jsonl"
test_path = cwd + "/subtaskA_test_monolingual.jsonl"
random_seed = 2024

train_df, val_df, test_df = get_data(train_path,test_path,random_seed)

print(train_df.head(3))
print()
print(test_df.head(3))
print()
print(cwd)
print()
print()
################################################################

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

model_checkpoint = ".\results"

batch_size = 16
print(model_checkpoint)

### 0 is human, 1 is machine ############################
id2label={
"0": "Human-Written",
"1": "Machine-Generated",
}
label2id={
"Human-Written": 0,
"Machine-Generated": 1,
}
###################################################################
fine_tune(train_df, val_df, checkpoints_path = model_checkpoint, id2label = id2label , label2id = label2id, model = model )

#best_model_path = 
#results,preds = test(test_df, model_path=best_model_path, id2label= id2label , label2id=label2id )













