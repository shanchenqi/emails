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
from keybert import KeyBERT
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
                        text.append(' '.join(content.split('_')))
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
            keywords_key_bert = getkeywords_key_bert([" ".join(text)])
            temp = keywords_key_bert
            dic[name] = temp
    df = pd.read_csv(file_name)
    for i in range(df.shape[0]):
        if df.iloc[i,0] in dic:# and len(dic[df.iloc[i,0]])>=1:

            df.loc[i,"key_bert"] = (str(dic[df.iloc[i,0]][0]))
    df.to_csv(file_name.split('.')[0]+"res3.csv",index=0)


def getkeywords_key_bert(text):
    #text = getemail(name)
    stopwords = stopwordslist("stopwords.txt")
    kw_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = kw_extractor.extract_keywords(text, keyphrase_ngram_range=(1,1), stop_words=stopwords, min_df =1, use_maxsum = True,use_mmr = True)
    print("Keywords of article", keywords)
    return keywords

if __name__ == '__main__':

    files = "./openwall_content"
    stopwordsfile = "./stopwords.txt"
    keywords_to_scv("./names.csv", stopwordsfile)




