from inverted_index_gcp_original import InvertedIndex, MultiFileReader
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import json
import pickle
import nltk
import re
import sys
import json
import math
from contextlib import closing
from collections import Counter, OrderedDict
import collections
import itertools
from itertools import islice, count, groupby
import os
from operator import itemgetter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
from google.cloud import storage
nltk.download('stopwords')

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text):
  """
  This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

  Parameters:
  -----------
  text: string , represting the text to tokenize.

  Returns:
  -----------
  list of tokens (e.g., list of tokens).
  """
  list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                    token.group() not in english_stopwords]
  return list_of_tokens


def read_posting_list(inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
      tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list


def get_posting_gen(index):
  """
  This function returning the generator working with posting list.

  Parameters:
  ----------
  index: inverted index
  """
  words, pls = zip(*index.posting_lists_iter())
  return words, pls

# read the index that in the pkl file from bucket.
bucket_name = '324677285ass3' 
client = storage.Client()
for blob in client.list_blobs(bucket_name, prefix='postings_gcp'):
  if not blob.name.endswith("index.pkl"):
    continue
  with blob.open("rb") as f:
    idx_text = pickle.load(f)

bucket_name = 'katya_niv_title_index'
# read the doc length dict
for blob in client.list_blobs(bucket_name):
  if not blob.name.endswith("doc_length.pickle"):
    continue
  with blob.open("rb") as f:
    DL_text = pickle.load(f)


for blob in client.list_blobs(bucket_name):
  if not blob.name.endswith("id_to_title_dict.pickle"):
    continue
  with blob.open("rb") as f:
    id_title_dict = pickle.load(f)


def calc_tfidf_query(query, index):
  epsilon = .0000001
  term_vector = list(index.term_total.keys())
  total_vocab_size = len(index.term_total)
  Q = {}
  counter = Counter(query)
  for token in query:
    if token in index.df.keys():  # avoid terms that do not appear in the index.
      tf = counter[token] / len(query)  # term frequency divded by the length of the query
      df = index.df[token]
      idf = math.log((len(DL_text)) / (df + epsilon), 10)  # smoothing  
      try:
        Q[token] = tf * idf
      except:
        pass
    else:
      Q[token] = 0
  return Q


def calc_cosine_sim(doc_id, sums_tfidf_dict, terms_ft_dict, Q):
  # mone:
  mone_sum = 0
  for term in Q.keys():
    if ((doc_id, term) in terms_ft_dict.keys()):
      mone_sum += Q[term] * terms_ft_dict[(doc_id, term)]
  # mechane
  if doc_id not in sums_tfidf_dict.keys():
    return 0
  sigma_doc = math.sqrt(sums_tfidf_dict[doc_id])
  sigma_Q = norm(np.array(list(Q.values())))
  cos_similarity = mone_sum / (sigma_doc * sigma_Q)
  return cos_similarity


def calc_tf_idf_in_runtime(words,pls,index):
  d={}
  for w, pl in zip(words,pls):
    for tup in pl:
      df = index.df[w]            
      idf = math.log((len(DL_text))/(df +0.000001),10) #smoothing
      d[(tup[0],w)]=tup[1]*idf
  return d

def calc_sums_tfidf_dict(terms_ft_dict, pls):

    d={}
    res=[]
    for ps in pls:
        if(len(ps) > 16000):
          candidates = [tup[0] for tup in ps if tup[1]>115]
          res+=candidates
        elif(16000 >= len(ps) > 12000):
          candidates = [tup[0] for tup in ps if tup[1]>95]
          res+=candidates
          #print("11")
        elif( 12000 >= len(ps) > 8000):
          candidates = [tup[0] for tup in ps if tup[1]>75]
          res+=candidates
          #print("22")
        elif(8000 >= len(ps) > 4000):
          candidates = [tup[0] for tup in ps if tup[1]>40]
          res+=candidates 
          #print("33")
        elif(4000 >= len(ps) > 2000):
          candidates = [tup[0] for tup in ps if tup[1]>25]
          res+=candidates 
          #print("44")
        else:
          candidates = [tup[0] for tup in ps if tup[1]>15]
          res+=candidates 
          #print("55")

    for key in list(set(res)):
        arr=np.array([tup[1]*tup[1] for tup in terms_ft_dict.items() if tup[0][0]==key])
        d[key]=sum(arr)
    return d

def answer_query(query, index, words, pls, N=100):
  results = []
  res = []
  terms_ft_dict=calc_tf_idf_in_runtime(words,pls,index)
  sums_tfidf_dict=calc_sums_tfidf_dict(terms_ft_dict, pls)
  for term in np.unique(query):
    Q = calc_tfidf_query(query, index)
    if term in words:
      for ps in pls:
        if(len(ps) > 16000):
          candidates = [tup for tup in ps if tup[1]>115]
          res+=candidates
        elif(16000 >= len(ps) > 8000):
          candidates = [tup for tup in ps if tup[1]>85]
          res+=candidates
          #print("11")
        elif(8000 >= len(ps) > 2000):
          candidates = [tup for tup in ps if tup[1]>30]
          res+=candidates 
          #print("33")
        else:
          candidates = [tup for tup in ps if tup[1]>15]
          res+=candidates 
          #print("55")

      list_of_doc = list(set(res))
      for doc_id, freq in list_of_doc:
        cosine_res = calc_cosine_sim(doc_id, sums_tfidf_dict, terms_ft_dict, Q)
        results.append((doc_id, cosine_res))
  sorted_results = sorted(list(set(results)), key= lambda x: x[1], reverse = True)[:N]
  final_results = [(tup[0], id_title_dict[tup[0]]) for tup in sorted_results]
  return final_results


TUPLE_SIZE = 6
def posting_lists_iter_gcp(index, w, locs):
  """ A generator that reads one posting list from disk and yields
      a (word:str, [(doc_id:int, tf:int), ...]) tuple.
  """
  with closing(MultiFileReader()) as reader:
    b = reader.read(locs, index.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(index.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
  return w, posting_list


# function for answering the query - called from frontend
def search_body_answer_backend(query):
  relevant_words = []
  ps = []
  words = ((idx_text.df.keys()))
  query = tokenize(query)
  for term in query:
    if term in words:
      locs = idx_text.posting_locs[term]
      w, p = posting_lists_iter_gcp(idx_text, term, locs)
      relevant_words.append(w)
      ps.append(p) 
  return answer_query(query, idx_text, relevant_words, ps)