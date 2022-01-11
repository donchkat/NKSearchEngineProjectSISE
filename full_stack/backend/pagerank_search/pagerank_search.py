import csv
import json
from google.cloud import storage

bucket_name = 'katya_niv_title_index' 
client = storage.Client()

#reading the file
with open('page_rank_file.csv', mode='r') as infile:
  reader = csv.reader(infile)
  pagerank_dict = {rows[0]:rows[1] for rows in reader}

def get_dict():
  return pagerank_dict

#answer query
def search_pagerank(query):
  l = []
  for elem in query:
    l.append((elem, pagerank_dict[str(elem)]))
  return l