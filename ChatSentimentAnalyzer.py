# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import timedelta
import nltk
from MessageCleaning import text_cleaner, str_cleaner 


csv_list = ['training_csv_1',
            'training_csv_2',
            'training_csv_n',
            'etc']

df_list = []

for csv in csv_list:
    df = pd.read_csv(f'{csv}.csv', index_col = 0)
    df_list.append(df)

df = pd.concat(df_list, axis=0, ignore_index=True)

df['sentiment'].value_counts()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timestamp'] = df['timestamp'].apply(lambda x: x - timedelta(hours=5))

df.dropna(axis=0, inplace=True)

# To train our model, we only need the text and sentiment columns 
text = df[['chat','sentiment']]

# We remove any usernames, hashtags, links, stopwords, and punctuation from our chat messages
# We also try to adjust the spelling on words such as 'soooooo' to 'soo'
# While not perfect, it reduces different variations such as 'sooo', 'soooooooo', etc.
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

stopwords = nltk.corpus.stopwords.words('english')
addi_sw = ['hi','hii','hai','haii','hey','heyy', 'he\'s', 'hes', 'im', 'shouldve', 'that\'s', 'thats', 'youd','youll','youre','youve', 'ur', 'u', 'r', 'uve', 'u\'ve']
stopwords.extend(addi_sw)
stopwords.remove("not")

wnl = WordNetLemmatizer()

text_cleaner(text)

# We stratify based on the Y column as the 'negative' class is under-represented 
# in comparison to the other two classes, so our train/test splits will have equal 
# values of each class
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tv = TfidfVectorizer(binary=False, ngram_range=(1, 3))
tv_lr = LogisticRegression(max_iter=200)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score


def lr_cv(X, Y, pipeline, splits=5, average_method='macro'):
    global fit_lr
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X,Y): 
        fit_lr = pipeline.fit(X.iloc[train],Y.iloc[train])
        predictions = fit_lr.predict(X.iloc[test])
        scores = fit_lr.score(X.iloc[test],Y.iloc[test])
        
        accuracy.append(scores * 100)
        precision.append(precision_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('LR            negative    neutral     positive')
        print('precision:',precision_score(Y.iloc[test], predictions, average = None))
        recall.append(recall_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('recall:   ',recall_score(Y.iloc[test], predictions, average = None))
        f1.append(f1_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('f1 score: ', f1_score(Y.iloc[test], predictions, average = None))
        print('-'*50)
    
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
    print("-------------")
    return fit_lr
    

# Conducted model comparisons between Logistic Regression and Naive Bayes
# Also within each model, tested between the cleaned dataset and SMOTE oversampling 
# for the minority class, but there was not a noticeable increase in average recall 
# in relation to the drop in average precision 

from sklearn.pipeline import Pipeline 

original_pipeline = Pipeline([
    ('vectorizer', tv),
    ('classifier', tv_lr)
    ])


lr_cv(text['cleaned'], text['sentiment'], original_pipeline)


from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

smt = SMOTE(random_state = 42, k_neighbors = 3)
SMOTE_pipeline = make_pipeline(tv, smt, tv_lr)

lr_cv(text['cleaned'], text['sentiment'], SMOTE_pipeline)


from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB()

def nb_cv(X, Y, pipeline, splits=5, average_method='macro'):
    global fit_nb
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X,Y): 
        fit_nb = pipeline.fit(X.iloc[train],Y.iloc[train])
        predictions = fit_nb.predict(X.iloc[test])
        scores = fit_nb.score(X.iloc[test],Y.iloc[test])
        
        accuracy.append(scores * 100)
        precision.append(precision_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('NB            negative    neutral     positive')
        print('precision:',precision_score(Y.iloc[test], predictions, average = None))
        recall.append(recall_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('recall:   ',recall_score(Y.iloc[test], predictions, average = None))
        f1.append(f1_score(Y.iloc[test], predictions, average = average_method) * 100)
        print('f1 score: ', f1_score(Y.iloc[test], predictions, average = None))
        print('-'*50)
    
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
    print("-------------")
    return fit_nb

nb_pipeline = Pipeline([
    ('vectorizer', tv),
    ('classifier', MNB)
    ])

nb_SMOTE_pipeline = make_pipeline(tv, smt, MNB)

nb_cv(text['cleaned'], text['sentiment'], nb_pipeline)

nb_cv(text['cleaned'], text['sentiment'], nb_SMOTE_pipeline)

# Looking at the confusion matrix scores, the first Logistic Regression model produced 
# accuracy, precision, and F1 score averages at or above 70%, with recall not far
# behind at 67%. 

# The underrepresented "Negative" class, has lower recall scores and in turn lower F1 scores
# while maintaining a high precision score. This means that the model is not sensitive 
# enough and is marking a high number of false negatives for the underrepresented class.

# We export our model to the 'fit_lr.pkl' file in order to be able to call it 
# within our TwitchBot file and perform classification for messages as they come in. 

import joblib

joblib.dump(fit_lr, 'fit_lr.pkl')


