import json
import pickle
from google.cloud import storage

bucket_name = 'katya_niv_title_index' 
client = storage.Client()

#reading the file with the page views
for blob in client.list_blobs(bucket_name):
    if not blob.name.endswith('pageviews-202108-user.pkl'):
        continue
    with blob.open("rb") as f:
        page_views_dict = pickle.load(f)

def get_dict():
  return page_views_dict

#answering query
def search_pageviews(query):
    l=[]
    for elem in query:
       l.append((elem,page_views_dict[elem]))
    return l