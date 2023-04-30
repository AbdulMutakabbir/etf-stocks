from pydantic import BaseModel
from enum import Enum

class ModelType(Enum):
    RL = "Random Forest"
    DL = "Deep Learning"

class PredictModel(BaseModel):
    vol_moving_avg: float
    adj_close_rolling_med: float
    model_type: ModelType = ModelType.RL
