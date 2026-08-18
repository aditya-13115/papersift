from fastapi import FastAPI
from app.endpoints.health import api_router as health_router
from app.endpoints.metrics import api_router as metrics_router
from app.endpoints.generate import api_router as generate_router
from app.endpoints.stream import api_router as stream_router
from app.core.exception_handlers import register_exception_handlers


app = FastAPI()


register_exception_handlers(app)


app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(generate_router)
app.include_router(stream_router)