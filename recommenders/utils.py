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


def get_recos_from_dict(user_id, recos:Dict[str, List[int]], k_recs=10):
    user_id = str(user_id)
    if user_id in recos:
        return recos[user_id][:k_recs]
    return get_popular(k_recs=k_recs)