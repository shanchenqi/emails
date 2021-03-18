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



def getkeywords_tfidf_sklearn(text):

    # text = getemail(name)
    # text = lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    vectorizer = CountVectorizer(max_df = 0.85, max_features=2000, stop_words='english')
    #该类会统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    #将文本转为词频矩阵并计算tf-idf
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(text))
    #将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # x_train_weight = tf_idf.toarray()
    feature_names = vectorizer.get_feature_names()
    doc = text[2]
    tf_idf_vector = tf_idf_transformer.transform(vectorizer.transform([doc]))
    sorted_item = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_item,10)
    # stopwords = args.stopwords
    print("tf-idf:", keywords.keys())
    return keywords

def stopwordslist(stopwordsfile):
    stopwords = [line.strip() for line in open(stopwordsfile, encoding='UTF-8').readlines()]
    return stopwords

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn+5]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        if str.isdigit(feature_names[idx]) and int(feature_names[idx])not in range(2008,2021):
            continue
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals[:topn])):
        results[feature_vals[idx]]=score_vals[idx]
    return results

def read_name_lists(files):
    df = pd.read_csv(files, usecols=[0], names=None, keep_default_na=False)  # 读取项目名称列,不要列名
    df_li = df.values.tolist()
    result = []
    for s_li in df_li:
        result.append(s_li[0])
    # print(result)
    return result

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

def getkeywords_sentence_transformer(text):
    stopwords = stopwordslist("./stopwords.txt")
    # print("len(stopwords)", len(stopwords))
     # Extract candidate words/phrases
    count = CountVectorizer( max_features=2000, stop_words=stopwords).fit(text)
    candidates = count.get_feature_names()
    model = SentenceTransformer(r'xlm-r-distilroberta-base-paraphrase-v1')
    doc_embedding = model.encode(text)
    candidate_embeddings = model.encode(candidates)
    n_gram_range = (1,1)
    top_n = 15
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    print("bert:", keywords)

    return keywords

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
    # getkeywords_lda("Kurt", stopwordsfile)
    # keywords_to_scv("/home/chenqi/work/openwall/names.csv", stopwordsfile, args.startDate, args.endDate)
    keywords_to_scv("./names.csv", stopwordsfile)
    # getemail("Kurt", "2008-10-10","2020-3-15")
    # date_list=getDateList(args.startDate, args.endDate)
    # getkeywords_sentence_transformer("Kurt")

    # print(date_list)




