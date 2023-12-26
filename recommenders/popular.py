import os
import json

import pandas as pd

from recommenders.model_loader import load

with open('recommenders/reco_paths.json') as jf:
    reco_paths = json.load(jf)
popularity_df = pd.read_csv(reco_paths['popular_csv'])


def get_popular(k_recs: int = 10):
    popular = popularity_df.head(k_recs).item_id.tolist()
    return popular
