import os
import random

from recommenders.lightfm import get_offline_recos_lightfm, get_recos_lightfm_ann
from recommenders.model_loader import load
from recommenders.neural_recommenders import get_recos_AE, get_recos_multi_VAE
from recommenders.popular import get_popular


def test_popular():
    k_recs = 10
    reco = get_popular(k_recs)
    assert isinstance(reco, list)
    assert len(reco) == k_recs


def test_userknn():
    MODEL_PATH = "models/user_knn.pkl"
    if os.path.exists(MODEL_PATH):
        userknn_model = load(MODEL_PATH)
        user_id = 0
        k_recs = 10
        reco = userknn_model.recommend(user_id, N_recs=k_recs)
        assert isinstance(reco, list)
        assert len(reco) == k_recs
    else:
        pass


def test_lightfm_ann():
    MODEL_PATH = "models/ANN_LightFM_warp_8.pkl"
    if os.path.exists(MODEL_PATH):
        user_id = random.randint(0, 10**9)
        k_recs = 10
        reco = get_recos_lightfm_ann(user_id, k_recs=k_recs)
        assert isinstance(reco, list)
        assert len(reco) == k_recs
    else:
        pass


def test_offline_lightfm():
    RECOS_PATH = "recommenders/offline/LightFM_warp_8.csv"
    if os.path.exists(RECOS_PATH):
        user_id = random.randint(0, 10**9)
        k_recs = 10
        reco = get_offline_recos_lightfm(user_id)
        assert isinstance(reco, list)
        assert len(reco) == k_recs
    else:
        pass


def test_neural():
    user_id = random.randint(0, 10**9)
    k_recs = 10
    for get_recos_fn in [get_recos_AE, get_recos_multi_VAE]:
        reco = get_recos_fn(user_id, k_recs=k_recs)
        assert isinstance(reco, list)
        assert len(reco) == k_recs
