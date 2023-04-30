from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, index
import joblib



app = FastAPI(
    title="Stocks/ETF Prediction",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    description="This swagger page contains documentation to access the model prediction API's"
)


# TODO: In PROD this should change
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Including the API routes
app.include_router(index.router)
app.include_router(predict.router, prefix="/api")


