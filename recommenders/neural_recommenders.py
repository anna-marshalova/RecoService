import json

from recommenders.utils import load_recos, get_recos_from_dict

with open('recommenders/reco_paths.json') as jf:
    reco_paths = json.load(jf)

AE_recos = load_recos(reco_paths['ae_json'])
multi_VAE_recos = load_recos(reco_paths['multi_vae_json'])

def get_recos_AE(user_id, k_recs=10):
    return get_recos_from_dict(user_id, recos=AE_recos, k_recs=k_recs)

def get_recos_multi_VAE(user_id, k_recs=10):
    return [int(r) for r in get_recos_from_dict(user_id, recos=multi_VAE_recos, k_recs=k_recs)]
    