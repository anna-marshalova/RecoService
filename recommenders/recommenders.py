from enum import Enum

from recommenders.popular import get_popular
from recommenders.userknn import get_recos_user_knn
from recommenders.lightfm import get_offline_recos_lightfm, get_recos_lightfm_ann
from recommenders.neural_recommenders import get_recos_AE, get_recos_multi_VAE
from recommenders.hybrid import get_recos_hybrid

class ModelName(str, Enum):
    popular = "popular"
    userknn = "userknn"
    lightfm = "lightfm"
    lightfm_ann = "lightfm_ann"
    autoencoder = "autoencoder"
    multi_vae = "multi_vae"
    hybrid = "hybrid"
    other = "unknown"


def get_popular_wrapper(user_id, k_recs=10):
    return get_popular(k_recs)
recommender_functions = {
    ModelName.popular: get_popular_wrapper,
    ModelName.userknn: get_recos_user_knn,
    ModelName.lightfm: get_offline_recos_lightfm,
    ModelName.lightfm_ann: get_recos_lightfm_ann,
    ModelName.autoencoder: get_recos_AE,
    ModelName.multi_vae: get_recos_multi_VAE,
    ModelName.hybrid: get_recos_hybrid
}