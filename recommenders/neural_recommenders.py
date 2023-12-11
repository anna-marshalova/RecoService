import os
import json
from typing import Dict, List

from recommenders.popular import get_popular

with open('recommenders/reco_paths.json') as jf:
    reco_paths = json.load(jf)

def load_recos(path):
    if os.path.exists(path):
        with open(path) as jf:
            recos = json.load(jf)
        return recos
    return {}

AE_recos = load_recos(reco_paths['ae_json'])
multi_VAE_recos = load_recos(reco_paths['multi_vae_json'])

def get_recos_from_dict(user_id, recos:Dict[str, List[int]], k_recs=10):
    user_id = str(user_id)
    if user_id in recos:
        return recos[user_id][:k_recs]
    return get_popular(k_recs=k_recs)

def get_recos_AE(user_id, k_recs=10):
    return get_recos_from_dict(user_id, recos=AE_recos, k_recs=k_recs)

def get_recos_multi_VAE(user_id, k_recs=10):
    return [int(r) for r in get_recos_from_dict(user_id, recos=multi_VAE_recos, k_recs=k_recs)]
    