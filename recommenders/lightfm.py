import os
import json
import pickle

import pandas as pd

from rectools import Columns
from rectools.tools import UserToItemAnnRecommender
from recommenders.popular import get_popular

with open('recommenders/reco_paths.json') as jf:
    reco_paths = json.load(jf)

lightfm_ann = None
MODEL_PATH = reco_paths['lightfm_ann_model']
if os.path.exists(MODEL_PATH):
    lightfm_ann = pickle.load(open(MODEL_PATH, 'rb'))


all_recos = pd.DataFrame([])
all_users = []
RECOS_PATH = reco_paths['lightfm_csv']
if os.path.exists(RECOS_PATH):
    all_recos = pd.read_csv(RECOS_PATH)
    all_users = all_recos[Columns.User].unique()

def get_offline_recos_lightfm(user_id, k_recs=10):
    if user_id in all_users:
        user_recos = all_recos[all_recos[Columns.User] == user_id][Columns.Item]
        return user_recos.tolist()[:k_recs]
    return get_popular(k_recs)

def get_recos_lightfm_ann(user_id, k_recs=10):
    if user_id in lightfm_ann.user_id_map.external_ids:
        return lightfm_ann.get_item_list_for_user(user_id, top_n=k_recs).tolist()
    else:
        return get_popular(k_recs)
    