import json

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

"""First read the file."""

with open("subtaskA_train_monolingual.jsonl", 'r', encoding='utf-8') as f:
    training_input=(f.read())
with open("subtaskA_monolingual.jsonl", 'r', encoding='utf-8') as f:
    testing_input=(f.read())

"""set the text and label into two lists."""

training_input=training_input.split("\n")
training_input = [json.loads(training_input[i])for i in range(len(training_input)) if len(training_input[i])>0]
training_text = [training_input[i]["text"]for i in range(len(training_input)) ]
training_label = [training_input[i]["label"]for i in range(len(training_input)) ]
testing_input=testing_input.split("\n")
testing_input = [json.loads(testing_input[i])for i in range(len(testing_input)) if len(testing_input[i])>0]
testing_text = [testing_input[i]["text"]for i in range(len(testing_input)) ]
testing_label = [testing_input[i]["label"]for i in range(len(testing_input)) ]

"""First we try the SVM to set the first baseline.
Initialize TF-IDF Vectorizer
"""

tfidf_vectorizer = TfidfVectorizer(max_features=1000)  #  adjust max_features as needed

"""Transform the training data"""

train_tfidf = tfidf_vectorizer.fit_transform(training_text)

"""Transform the test data"""

test_tfidf = tfidf_vectorizer.transform(testing_text)

"""Initialize and train SVM classifier. This process may take up to 3 hours."""

logistic_classifier = LogisticRegression()
logistic_classifier.fit(train_tfidf, training_label)

"""Predict on the test data, this step may take up to half an hour."""

predict_label = logistic_classifier.predict(test_tfidf)

"""Calculate accuracy"""

accuracy = accuracy_score(testing_label, predict_label)
print("Accuracy:", accuracy)

"""Write a predicted label."""

predict_label=predict_label.tolist()
with open("logistic_label.jsonl", 'w', encoding='utf-8') as f:
  for i in range(len(predict_label)):
    json.dump(predict_label[i], f)
    f.write('\n')

""" The downstream work can start here to save some time."""

#!pip install datasets
from datasets import Dataset
import pandas as pd

with open("logistic_label.jsonl", 'r', encoding='utf-8') as f:
    result=(f.read())
result=[result]
result=result[0].split("\n")
label=[]
id=[]
for i in range(len(result)-1):
  label.append(int(result[i]))
  id.append(i)

"""Generate the output has the same format as the input."""

predictions_df = pd.DataFrame({'id': id, 'label': label})
predictions_df.to_json("logistic.jsonl", lines=True, orient='records')

with open("logistic.jsonl", 'r', encoding='utf-8') as f:
    result=(f.read())

result
