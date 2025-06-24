from fastapi import FastAPI
from routes import predict

app = FastAPI(title="fastapi")

app.include_router(predict.router, tags=["predict"])

