import transformer_baseline
from transformer_baseline import get_data
import os

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import spacy
import numpy as np
import pandas as pd

cwd = os.getcwd()
print(cwd)

train_path = cwd+ "/subtaskA_train_monolingual.jsonl"
test_path =  cwd + "/subtaskA_test_monolingual.jsonl"
print(train_path)
print(test_path)

random_seed = 2024

train_df, val_df, test_df = get_data(train_path,test_path,random_seed)


print(train_df.head(3))
print()
print(test_df.head(3))


### 0 is human, 1 is machine #############################

# Train Naïve Bayes classifier

# Load the dataset
tfidf_vectorizer = TfidfVectorizer(max_features=1000)

X_train = tfidf_vectorizer.fit_transform(train_df['text'])
Y_train = train_df['label']

X_test = tfidf_vectorizer.transform(test_df['text'])

# NB #####################
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, Y_train)

# Evaluate classifiers
nb_prediction = nb_classifier.predict(X_test)

print("successfully predicted")

Resulted_test = pd.concat([test_df['id'], pd.DataFrame(data = nb_prediction, columns = ['label'])], axis = 1)

print("saving result successfully, now writing to .json")

result_path = cwd+'/pred_nb.json'

print(result_path)

Resulted_test.to_json(result_path, orient='values')



