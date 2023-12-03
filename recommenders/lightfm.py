import os
import pickle
import pandas as pd
from rectools import Columns
from rectools.tools import UserToItemAnnRecommender
from recommenders.popular import get_popular


all_recos = pd.read_csv('recommenders/offline/LightFM_warp_8.csv')
all_users = all_recos[Columns.User].unique()
popular_recos = pd.read_csv(os.path.join('kion_train', "popular_50.csv")).item_id.tolist()
lightfm_ann = pickle.load(open("models/ANN_LightFM_warp_8.pkl", 'rb'))

def get_offline_recos_lightfm(user_id):
    if user_id in all_users:
        user_recos = all_recos[all_recos[Columns.User] == user_id][Columns.Item]
        return user_recos.tolist()
    return popular_recos[:10]

def get_recos_lightfm_ann(user_id, k_recs=10):
    if user_id in lightfm_ann.user_id_map.external_ids:
        return lightfm_ann.get_item_list_for_user(user_id, top_n=k_recs).tolist()
    else:
        return get_popular(k_recs)
    