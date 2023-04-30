from fastapi import APIRouter, HTTPException
from api.model.predict_model import ModelType
import joblib 

router = APIRouter()

model = {}
# model[ModelType.RF] = joblib.load("rf.joblib")
# model[ModelType.DL] = joblib.load("dl.joblib")

@router.get(
        "/predict", 
        tags=['Prediction'],
        description='''This endpoint will return prediction for the pre-trained models. 
        By default the model is set to "Random Forest". 
        Two input features (vol_moving_avg, adj_close_rolling_med) need to be provide to the model to make predictions.'''
    )
async def predict(vol_moving_avg:float, adj_close_rolling_med:float, model_type:ModelType=ModelType.DL)->int:
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
    prediction = model[model_type].predict()
    return prediction
