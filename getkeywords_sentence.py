#!/usr/bin/env python
# coding=utf-8
import jieba.analyse
import os, sys
from nltk.corpus import stopwords
import argparse

import numpy as np
import pandas as pd
import re, nltk, spacy, gensim
#from rake_nltk import Rake
from gensim import corpora, models, similarities
# from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import datetime

def getDateList(start_date, end_date):
    date_list = []
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    date_list.append(start_date.strftime('%Y-%m-%d'))
    while start_date <= end_date:
        start_date += datetime.timedelta(days=1)
        date_list.append(start_date.strftime('%Y-%m-%d'))
    return date_list

def getemail(name, date1="2008-1-1", date2="2020-12-12"):
    date_list=getDateList(date1, date2)
    text = []
    for date in date_list:
        year = date.split("-")[0]
        month = date.split("-")[1]
        day = date.split("-")[2]

        if os.path.exists(os.path.join(files, year, month, day)):
            for idex in os.listdir(os.path.join(files, year, month, day)):
                with open(os.path.join(files, year, month, day, idex), 'r') as txtfile:
                    content = txtfile.readline()
                    if re.findall(name, content):
                        text.append(content)
    print("get %d emails"%len(text))
    return text



def stopwordslist(stopwordsfile):
    stopwords = [line.strip() for line in open(stopwordsfile, encoding='UTF-8').readlines()]
    return stopwords


def keywords_to_scv(file_name, stopwordsfile, date1="2008-1-1", date2="2020-12-12"):
    names = read_name_lists(file_name)
    dic = {}
    for name in names:
        if not name in dic and len(name)>0:
            print("name:",name)
            text = getemail(name,date1, date2)
            keywords_sentence_transformer = getkeywords_sentence_transformer(text)
            dic[name] = keywords_sentence_transformer
    df = pd.read_csv(file_name)
    for i in range(df.shape[0]):
        if df.iloc[i,0] in dic:
            df.loc[i,"bert"] = str(dic[df.iloc[i,0]])
    df.to_csv(file_name.split('.')[0]+"res4.csv",index=0)

def getkeywords_sentence_transformer(text):
    stopwords = stopwordslist("./stopwords.txt")
     # Extract candidate words/phrases
    count = CountVectorizer(max_df = 0.85, max_features=2000, stop_words=stopwords).fit(text)
    candidates = count.get_feature_names()
    model = SentenceTransformer(r'xlm-r-distilroberta-base-paraphrase-v1')
    doc_embedding = model.encode(text)
    candidate_embeddings = model.encode(candidates)
    n_gram_range = (1,1)
    top_n = 10
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    num_regex = re.compile(r'[0-9]')
    keywords = []
    for index in distances.argsort()[0][-top_n:]:
        if len(num_regex.findall(candidates[index]))<=1:
            keywords.append(candidates[index]+':'+str(distances[0][index]))

    print("bert:", keywords[::-1])

    return keywords[::-1]
if __name__ == '__main__':
    files = "./openwall_content"
    stopwordsfile = "./stopwords.txt"
    keywords_to_scv("./names.csv", stopwordsfile)




