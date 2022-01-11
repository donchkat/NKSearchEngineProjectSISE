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
  print(query)
  for term in np.unique(query):
    Q = calc_tfidf_query(query, index)
    if term in words:
      for ps in pls:
        if(len(ps) > 16000):
          candidates = [tup for tup in ps if tup[1]>115]
          res+=candidates
        elif(16000 >= len(ps) > 12000):
          candidates = [tup for tup in ps if tup[1]>95]
          res+=candidates
          #print("11")
        elif( 12000 >= len(ps) > 8000):
          candidates = [tup for tup in ps if tup[1]>75]
          res+=candidates
          #print("22")
        elif(8000 >= len(ps) > 4000):
          candidates = [tup for tup in ps if tup[1]>40]
          res+=candidates 
          #print("33")
        elif(4000 >= len(ps) > 2000):
          candidates = [tup for tup in ps if tup[1]>25]
          res+=candidates 
          #print("44")
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
def search_answer_backend(query):
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



#THE FIRST TRY
#==============================================================================#
# from inverted_index_gcp_original import InvertedIndex, MultiFileReader
# import nltk
# import re
# import sys
# import json
# from numpy import dot
# from numpy.linalg import norm
# import math
# from contextlib import closing
# from collections import Counter, OrderedDict
# import collections
# import itertools
# from itertools import islice, count, groupby
# import pandas as pd
# import os
# from operator import itemgetter
# from nltk.stem.porter import *
# from nltk.corpus import stopwords
# from time import time
# from timeit import timeit
# from pathlib import Path
# import pickle
# import numpy as np
# from google.cloud import storage
# from itertools import chain
# import builtins
# # #import body_search
# # import title_search
# # import pagerank_search
# # import pageviews_search
# nltk.download('stopwords')



# TUPLE_SIZE = 6
# TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

# english_stopwords = frozenset(stopwords.words('english'))
# corpus_stopwords = ["category", "references", "also", "external", "links",
#                     "may", "first", "see", "history", "people", "one", "two",
#                     "part", "thumb", "including", "second", "following",
#                     "many", "however", "would", "became"]

# all_stopwords = english_stopwords.union(corpus_stopwords)
# RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

# #stemmer=PorterStemmer()
# def tokenize(text):
#   """
#   This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

#   Parameters:
#   -----------
#   text: string , represting the text to tokenize.

#   Returns:
#   -----------
#   list of tokens (e.g., list of tokens).
#   """
#   list_of_tokens =  [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]

#   return list_of_tokens


# def read_posting_list(inverted, w):
#   with closing(MultiFileReader()) as reader:
#     locs = inverted.posting_locs[w]
#     b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
#     posting_list = []
#     for i in range(inverted.df[w]):
#       doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
#       tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
#       posting_list.append((doc_id, tf))
#     return posting_list


# def get_posting_gen(index):
#   """
#   This function returning the generator working with posting list.

#   Parameters:
#   ----------
#   index: inverted index
#   """
#   words, pls = zip(*index.posting_lists_iter())
#   return words, pls


# # read the index that in the pkl file from bucket.
# #WORKS FOR THE SIMPLE METHOD WE DID
# # bucket_name = 'katya_niv_title_index' 
# # client = storage.Client()
# # for blob in client.list_blobs(bucket_name, prefix='postings_gcp_search'):
# #   if not blob.name.endswith("index_search.pkl"):
# #     continue
# #   with blob.open("rb") as f:
# #     idx_text = pickle.load(f)


# #TRYING TO USE THE ORIGINAL INDEX AND FILE
# bucket_name = '324677285ass3' 
# client = storage.Client()
# for blob in client.list_blobs(bucket_name, prefix='postings_gcp'):
#   if not blob.name.endswith("index.pkl"):
#     continue
#   with blob.open("rb") as f:
#     idx_text = pickle.load(f)


# bucket_name = 'katya_niv_title_index' 
# for blob in client.list_blobs(bucket_name):
#   if not blob.name.endswith("doc_length.pickle"):
#     continue
#   with blob.open("rb") as f:
#     DL_text = pickle.load(f)

# for blob in client.list_blobs(bucket_name):
#   if not blob.name.endswith("id_to_title_dict.pickle"):
#     continue
#   with blob.open("rb") as f:
#     id_title_dict = pickle.load(f)


# TUPLE_SIZE = 6
# def posting_lists_iter_gcp(index, w, locs):
#   """ A generator that reads one posting list from disk and yields
#       a (word:str, [(doc_id:int, tf:int), ...]) tuple.
#   """
#   with closing(MultiFileReader()) as reader:
#     b = reader.read(locs, index.df[w] * TUPLE_SIZE)
#     posting_list = []
#     for i in range(index.df[w]):
#       doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
#       tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
#       posting_list.append((doc_id, tf))
#   return w, posting_list


# def get_candidate_documents(query_to_search, index, words, pls):
#   """
#   Generate a dictionary representing a pool of candidate documents for a given query.

#   Parameters:
#   -----------
#   query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
#                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

#   index:           inverted index loaded from the corresponding files.

#   words,pls: generator for working with posting.
#   Returns:
#   -----------
#   list of candidates. In the following format:
#                                                               key: pair (doc_id,term)
#                                                               value: tfidf score.
#   """
#   candidates = []
#   res=[]
#   # for term in np.unique(query_to_search):
#   #   if term in words:        
#   #     current_list = (pls[words.index(term)])                
#   #     candidates += current_list    
#   # return np.unique(candidates)

#   # doc_id_lst = []
#   # res=[]
#   # for term in query_to_search:
#   #   if term in words:

#   #candidates = [ps[0] for ps in pls]
#   for ps in pls:
#     if(len(ps) > 16000):
#       candidates = [tup[0] for tup in ps if tup[1]>115]
#       res+=candidates
#     elif(16000 >= len(ps) > 12000):
#       candidates = [tup[0] for tup in ps if tup[1]>95]
#       res+=candidates
#       #print("11")
#     elif( 12000 >= len(ps) > 8000):
#       candidates = [tup[0] for tup in ps if tup[1]>75]
#       res+=candidates
#       #print("22")
#     elif(8000 >= len(ps) > 4000):
#       candidates = [tup[0] for tup in ps if tup[1]>40]
#       res+=candidates 
#       #print("33")
#     elif(4000 >= len(ps) > 2000):
#       candidates = [tup[0] for tup in ps if tup[1]>25]
#       res+=candidates 
#       #print("44")
#     else:
#       candidates = [tup[0] for tup in ps if tup[1]>15]
#       res+=candidates 
#       #print("55")
#   return list(set(res))

#   #   candidates += curr_ps
#   # doc_id_lst= [tup[0] for tup in candidates]
#   # return list(set(doc_id_lst))

# # When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
# class BM25_from_index:
#   """
#   Best Match 25.
#   ----------
#   k1 : float, default 1.5

#   b : float, default 0.75

#   index: inverted index
#   """
#   DL = DL_text

#   def __init__(self, index, k1=1.2, b=0.75):
#     self.b = b
#     self.k1 = k1
#     self.index = index
#     self.N = len(DL_text)
#     self.AVGDL = builtins.sum(list(DL_text.values())) / self.N

#   def calc_idf(self, list_of_tokens):
#     """
#     This function calculate the idf values according to the BM25 idf formula for each term in the query.

#     Parameters:
#     -----------
#     query: list of token representing the query. For example: ['look', 'blue', 'sky']

#     Returns:
#     -----------
#     idf: dictionary of idf scores. As follows:
#                                                 key: term
#                                                 value: bm25 idf score
#     """
#     idf = {}
#     for term in list_of_tokens:
#       if term in self.index.df.keys():
#         n_ti = self.index.df[term]
#         idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
#       else:
#         pass
#     return idf

#   def search(self, query, N=100):
#     """
#     This function calculate the bm25 score for given query and document.
#     We need to check only documents which are 'candidates' for a given query.
#     This function return a dictionary of scores as the following:
#                                                                 key: query_id
#                                                                 value: a ranked list of pairs (doc_id, score) in the length of N.

#     Parameters:
#     -----------
#     dictionary
#     query: list of token representing the query. For example: ['look', 'blue', 'sky']
#     doc_id: integer, document id.

#     Returns:
#     -----------
#     score: float, bm25 score.
#     """
#     # YOUR CODE HERE
#     d = {}
#     relevant_words = []
#     ps = []
#     my_words = (self.index.df.keys())
#     #print("query is: " + str(query))
#     #print(len(my_words))
#     for term in query:
#       #print(term in my_words)
#       if term in my_words:
#         #print("Term in words! "+str(term))
#         locs = self.index.posting_locs[term]
#         #print("1")
#         w, p = posting_lists_iter_gcp(self.index, term, locs)
#         #print("2")
#         relevant_words.append(w)
#         ps.append(p)
#     #print(ps)
#     #print(ps)
#     #print(relevant_words)
#     self.words, self.pls = relevant_words, ps
#     self.idf = self.calc_idf(query)

#     mydict = get_candidate_documents(query, self.index, self.words, self.pls)
#     #print(mydict)
#     doc_score_lst =[]
#     for doc in mydict:
#       if doc in DL_text.keys():
#         #print("3")
#         doc_score = self._score(query, doc)
#         #print("4")
#         doc_score_lst.append((doc, doc_score))
#     sorted_doc_score_lst = list(sorted(doc_score_lst, key=lambda x: x[1], reverse=True))[:N]
#     result = [(int(tup[0]), id_title_dict[tup[0]])for tup in sorted_doc_score_lst]
#     return result

#   def _score(self, query, doc_id):
#     """
#     This function calculate the bm25 score for given query and document.

#     Parameters:
#     -----------
#     query: list of token representing the query. For example: ['look', 'blue', 'sky']
#     doc_id: integer, document id.

#     Returns:
#     -----------
#     score: float, bm25 score.
#     """
#     candidates=[]
#     res=[]
#     score = 0.0
#     doc_len = DL_text[doc_id]
#     for term in query:
#       if term in self.index.df.keys():
#         for ps in self.pls:
#           if(len(ps) > 16000):
#             candidates = [tup for tup in ps if tup[1]>115]
#             res+=candidates
#           elif(16000 >= len(ps) > 12000):
#             candidates = [tup for tup in ps if tup[1]>95]
#             res+=candidates
#             #print("11")
#           elif( 12000 >= len(ps) > 8000):
#             candidates = [tup for tup in ps if tup[1]>75]
#             res+=candidates
#             #print("22")
#           elif(8000 >= len(ps) > 4000):
#             candidates = [tup for tup in ps if tup[1]>40]
#             res+=candidates 
#             #print("33")
#           elif(4000 >= len(ps) > 2000):
#             candidates = [tup for tup in ps if tup[1]>25]
#             res+=candidates 
#             #print("44")
#           else:
#             candidates = [tup for tup in ps if tup[1]>15]
#             res+=candidates 
#             #print("55")
#         term_frequencies = dict(list(set(res)))
#         if doc_id in term_frequencies.keys():
#           freq = term_frequencies[doc_id]
#           numerator = self.idf[term] * freq * (self.k1 + 1)
#           denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
#           score += (numerator / denominator)
#     return score



# def search_answer_backend(query):
#   bm25_text = BM25_from_index(idx_text)
#   bm25_queries_score_train = bm25_text.search(tokenize(query))

#   # res_title = title_search.search_title_answer_backend(query)[:100]
#   # res = sorted([(tup[0], tup[1], float(pagerank_search.get_dict()[str(tup[0])]), pageviews_search.get_dict()[tup[0]]) for tup in res_title], key=lambda x: 0.7*x[2]+0.3*x[3], reverse=True)
#   # sorted_res = [(t[0], t[1])for t in res]
#   # return sorted_res

#   return bm25_queries_score_train






