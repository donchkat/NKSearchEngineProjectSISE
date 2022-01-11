from inverted_index_gcp_at import InvertedIndex, MultiFileReader
import nltk
import re
import sys
import json
from numpy import dot
from numpy.linalg import norm
import math
from contextlib import closing
from collections import Counter, OrderedDict
import collections
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
from operator import itemgetter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
import pickle
import numpy as np
from google.cloud import storage

nltk.download('stopwords')

# backend file 
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


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

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


def posting_lists_iter_gcp(index, w, locs):
  """ A generator that reads one posting list from disk and yields
      a (word:str, [(doc_id:int, tf:int), ...]) tuple.
  """
  with closing(MultiFileReader()) as reader:
    b = reader.read(locs, index.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(index.df[w]):
      doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
      tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
  return w, posting_list


# read the index that in the pkl file from bucket.
bucket_name = 'katya_niv_title_index'
client = storage.Client()
for blob in client.list_blobs(bucket_name, prefix='anchor_text_index/postings_gcp'):
  if not blob.name.endswith("index_at.pkl"):
    continue
  with blob.open("rb") as f:
    idx_titles = pickle.load(f)


for blob in client.list_blobs(bucket_name):
  if not blob.name.endswith('id_to_title_dict.pickle'):
    continue
  with blob.open("rb") as f:
    id_titles_dict = pickle.load(f)



# candidate function for titles
def get_candidate_documents_titles(query_to_search, index, words, pls):
  """
  Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
  and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
  Then it will populate the dictionary 'candidates.'
  For calculation of IDF, use log with base 10.
  tf will be normalized based on the length of the document.

  Parameters:
  -----------
  query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                    Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

  index:           inverted index loaded from the corresponding files.

  words,pls: generator for working with posting.
  Returns:
  -----------
  dictionary of candidates. In the following format:
                                                              key: pair (doc_id,term)
                                                              value: tfidf score.
  """
  dict_docs_and_terms = {}
  for term in np.unique(query_to_search):
    if term in words:
      list_of_doc = pls[words.index(term)]
      for doc_id, freq in list_of_doc:
        dict_docs_and_terms[(doc_id, term)] = 1
  counts = collections.Counter(k[0] for k in dict_docs_and_terms)
  final_result = [(tup[0], id_titles_dict[tup[0]]) for tup in counts.most_common()]
  return final_result

# function for answering the query - called from frontend
def search_anchor_text_answer_backend(curr_query):
  relevant_words = []
  ps = []
  words = (idx_titles.term_total.keys())
  curr_query = tokenize(curr_query)
  for term in curr_query:
    if term in words:
      locs = idx_titles.posting_locs[term]
      w, p = posting_lists_iter_gcp(idx_titles, term, locs)
      relevant_words.append(w)
      ps.append(p)
  return get_candidate_documents_titles(curr_query, idx_titles, relevant_words, ps)
