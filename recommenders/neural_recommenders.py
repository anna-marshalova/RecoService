import os
import json
from typing import Dict, List

from recommenders.popular import get_popular


def load_recos(path):
    if os.path.exists(path):
        with open(path) as jf:
            recos = json.load(jf)
        return recos
    return {}

AE_recos = load_recos('recommenders/offline/AE_recos.json')
multi_VAE_recos = load_recos('recommenders/offline/MultiVAE_recos.json')

def get_recos_from_dict(user_id, recos:Dict[str, List[int]], k_recs=10):
    user_id = str(user_id)
    if user_id in recos:
        return recos[user_id][:k_recs]
    return get_popular(k_recs=k_recs)

def get_recos_AE(user_id, k_recs=10):
    return get_recos_from_dict(user_id, recos=AE_recos, k_recs=k_recs)

def get_recos_multi_VAE(user_id, k_recs=10):
    return [int(r) for r in get_recos_from_dict(user_id, recos=multi_VAE_recos, k_recs=k_recs)]
    