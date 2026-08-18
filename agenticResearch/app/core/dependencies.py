import asyncio
from app.services.circuit_breaker import CircuitBreaker
from app.services.model_router import SemanticRouter
from app.services.metrics_store import MetricsStore


metrics_store = MetricsStore(max_size=1000)
semaphore = asyncio.Semaphore(10)
circuit = CircuitBreaker(failure_threshold=5,recovery_timeout=30)
router = SemanticRouter()