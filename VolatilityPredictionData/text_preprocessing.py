# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 23:51:16 2024

@author: kkksk
"""
import re
import pandas as pd
import numpy as np

def str_to_array(s):
    list_array = [float(x) for x in s.strip('[]').split()]
    return np.array(list_array)

def text_preprocess(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text
    
def preprocess(file) :
    df = pd.read_csv(file)
    df['title'] = df['title'].apply(text_preprocess)
    df['embedding'] = df['embedding'].apply(str_to_array)
    
    return df


preprocessed_newsdata = preprocess('newsdata_embedding.csv')
preprocessed_newsdata.head(5)
preprocessed_newsdata.to_parquet('newsdata_preprocessed+embedding.parquet',
                                 engine = 'pyarrow',
                                 compression = 'gzip'
                                 )


df = pd.read_parquet('newsdata_preprocessed+embedding.parquet')
df.head()
df['embedding']

preprocessed_newsdata['embedding'][0]
