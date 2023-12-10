from enum import Enum

class ModelName(str, Enum):
    range = "range"
    popular = "popular"
    userknn = "userknn"
    lightfm = "lightfm"
    lightfm_ann = "lightfm_ann"
    autoencoder = "autoencoder"
    multi_vae = "multi_vae"
    other = "unknown"