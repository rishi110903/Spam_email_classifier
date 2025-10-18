import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("spam_ham_dataset.csv")
data = df.where(pd.notnull(df), '')
data.loc[data['label'] == "spam", 'label'] = 0
data.loc[data['label'] == "ham", 'label'] = 1
x = data['text']
y = data['label']
#test size = 0.2 is to train data 80% and 20% test, random state is to control randomness to ensure consistent result
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 3)
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase= True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

model = LogisticRegression()
model.fit(x_train_features, y_train)

#Accuracy Testing on trained data
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print("Accuracy on training data : ", accuracy_on_training_data)

#Accuracy Testing on testing data
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print("Accuracy on test data : ", accuracy_on_test_data)
#here i just pasted a spam mail text to get result in CLI
input_your_mail = ["this is a follow up to the note i gave you on monday 4 / 3 / 00 preliminary flow data provided by daren. please override pops daily volume presently zero to reflect daily"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)