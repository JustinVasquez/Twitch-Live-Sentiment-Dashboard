# -*- coding: utf-8 -*-

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

stopwords = nltk.corpus.stopwords.words('english')
addi_sw = ['hi','hii','hai','haii','hey','heyy', 'he\'s', 'hes', 'im', 'shouldve', 'that\'s', 'thats', 'youd','youll','youre','youve', 'ur', 'u', 'r', 'uve', 'u\'ve']
stopwords.extend(addi_sw)
stopwords.remove("not")
wnl = WordNetLemmatizer()

def text_cleaner(df):
    df['no_users_links_punct'] = df['chat'].apply(lambda x: re.sub('(#)|(@)([A-Za-z0-9_-]+\s?)|[0-9]|(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*?)|([^\w\s]|[_])+','',x.lower()))
    df['spell_corrected'] = df['no_users_links_punct'].apply(lambda x: re.sub(r'(.)\1+', r'\1\1', x))
    df['cleaned'] = df['spell_corrected'].apply(lambda x: ' '.join([wnl.lemmatize(word, pos='v') for word in word_tokenize(x) if word not in stopwords]))
    return df


def str_cleaner(text):
    wnl = WordNetLemmatizer()
    no_users_links_punct = re.sub('(#)|(@)([A-Za-z0-9_-]+\s?)|[0-9]|(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*?)|([^\w\s]|[_])+','',text.lower())
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', no_users_links_punct)
    cleaned = ' '.join([wnl.lemmatize(word, pos='v') for word in word_tokenize(spell_corrected) if word not in stopwords])
    return cleaned