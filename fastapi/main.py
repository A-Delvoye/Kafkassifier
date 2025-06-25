from routes import predict

from fastapi import FastAPI

app = FastAPI(title="fastapi")

app.include_router(predict.router, tags=["predict"])

