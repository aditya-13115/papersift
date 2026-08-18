from fastapi import APIRouter
from app.core.dependencies import metrics_store


api_router = APIRouter()

@api_router.get("/metrics")
async def metrics():
    return metrics_store.summary()