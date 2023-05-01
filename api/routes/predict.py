from fastapi import APIRouter, HTTPException
from api.model.predict_model import ModelType
from api.utils.DLModel import DLModel
import joblib 
import torch

router = APIRouter()

dl_model_stats = joblib.load("./models/dataset_stats_deep_learning.joblib")
models = {}
models[ModelType.RF] = joblib.load("./models/model_random_forest.joblib")
models[ModelType.DL] = DLModel(stats=dl_model_stats)
models[ModelType.DL].load_state_dict(torch.load("./models/model_deep_learning.joblib"))

@router.get(
        "/predict", 
        tags=['Prediction'],
        description='''This endpoint will return prediction for the pre-trained models. 
        By default the model is set to "Random Forest". 
        Two input features (vol_moving_avg, adj_close_rolling_med) need to be provide to the model to make predictions.'''
    )
async def predict(vol_moving_avg:float, adj_close_rolling_med:float, model_type:ModelType=ModelType.RF)->int:
    if vol_moving_avg < 0:
        raise HTTPException(
            status_code=422, 
            detail="vol_moving_avg should be greater than 0 (zero)."
        )
    if adj_close_rolling_med < 0:
        raise HTTPException(
            status_code=422,
            detail="adj_close_rolling_med should be greater than 0 (zero)."
        )
    X = [[vol_moving_avg, adj_close_rolling_med]]
    model = models[model_type]
    prediction = None
    if model_type == ModelType.RF:
        prediction = model.predict(X)[0]
    elif model_type == ModelType.DL:
        prediction = model.predict(
            vol_moving_avg=vol_moving_avg,
            adj_close_rolling_med=adj_close_rolling_med
        )[0]
    prediction = int(prediction)
    return prediction
