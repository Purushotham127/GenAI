from fastapi import FastAPI
from routes.mongodb_routes import router as mongodb_router

app = FastAPI()

app.include_router(mongodb_router)