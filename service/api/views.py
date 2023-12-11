from typing import List

from fastapi import APIRouter, FastAPI, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from recommenders.lightfm import get_offline_recos_lightfm, get_recos_lightfm_ann
from recommenders.model_names import ModelName
from recommenders.neural_recommenders import get_recos_AE, get_recos_multi_VAE
from recommenders.popular import get_popular
from recommenders.userknn import get_recos_user_knn
from service.api.exceptions import AuthorizationError, ModelNotFoundError, UserNotFoundError
from service.api.keys import API_KEYS
from service.log import app_logger


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
api_key_header = APIKeyHeader(name="Authorization")


def get_token(token: str = Security(api_key_header)) -> str:
    if token in API_KEYS:
        return token
    raise AuthorizationError()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(path="/reco/{model_name}/{user_id}", tags=["Recommendations"], response_model=RecoResponse)
async def get_reco(
    request: Request, model_name: ModelName, user_id: int, token: str = Security(get_token)
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")
    k_recs = request.app.state.k_recs

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")
    if model_name is ModelName.range:
        reco = list(range(k_recs))
    elif model_name is ModelName.popular:
        reco = get_popular(k_recs)
    elif model_name is ModelName.userknn:
        reco = get_recos_user_knn(user_id, k_recs=k_recs)
    elif model_name is ModelName.lightfm:
        reco = get_offline_recos_lightfm(user_id)
    elif model_name is ModelName.lightfm_ann:
        reco = get_recos_lightfm_ann(user_id, k_recs=k_recs)
    elif model_name is ModelName.autoencoder:
        reco = get_recos_AE(user_id, k_recs=k_recs)
    elif model_name is ModelName.multi_vae:
        reco = get_recos_multi_VAE(user_id, k_recs=k_recs)
    else:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
