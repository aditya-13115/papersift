from fastapi import APIRouter
from time import perf_counter
from app.schemas.requests import GenerateRequest
from app.services.llm_client import AsyncLLMClient
from app.core.dependencies import router, semaphore, circuit, metrics_store
from app.core.config import PRICING

api_router = APIRouter()

@api_router.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    start_time = perf_counter()
    model = router.route(request.prompt)
    print(f"ROUTED MODEL: {model}")
    client = AsyncLLMClient(model=model, semaphore=semaphore, circuit=circuit, max_completion_tokens=256)  # You can adjust the model and parameters as needed
    response = await client.generate(request.prompt)

    latency = perf_counter() - start_time

    usage = response["usage"]


    output_tokens_per_second = (usage.completion_tokens / usage.completion_time 
                                if usage.completion_time > 0
                                else 0)
    cost = (
        usage.prompt_tokens / 1_000_000 * PRICING[model]["input"]
        + usage.completion_tokens / 1_000_000 * PRICING[model]["output"]
    )

    metadata = {
        "model": model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,

        "queue_time": usage.queue_time,
        "prompt_time": usage.prompt_time,
        "completion_time": usage.completion_time,
        "total_time": usage.total_time,
        "cost": cost,
        "output_tokens_per_second": output_tokens_per_second,
    }
    # Store request metrics for /metrics endpoint
    metrics_store.record({
        "model": model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency": latency,
        "cost": cost,
    })

    return {
        "result": response["text"],
        "latency": latency,
        "model": model,
        "metadata": metadata
    }