from fastapi import APIRouter
from time import perf_counter
import asyncio
import groq
from collections.abc import AsyncIterable
from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from app.schemas.requests import GenerateRequest
from app.services.llm_client import AsyncLLMClient
from app.core.dependencies import router, semaphore, circuit, metrics_store
from app.core.config import PRICING

api_router = APIRouter()


@api_router.post("/stream", response_class=EventSourceResponse)
async def stream_endpoint(request: GenerateRequest) -> AsyncIterable[ServerSentEvent]:
    start_time = perf_counter()
    model = router.route(request.prompt)
    print(f"ROUTED MODEL: {model}")
    client = AsyncLLMClient(model=model, semaphore=semaphore, circuit=circuit, streaming=True, max_completion_tokens=256)  # You can adjust the model and parameters as needed
    completed = False
    try:
            async for token in client.stream(request.prompt):
                yield ServerSentEvent(data=token,event="token")
            completed = True

    except asyncio.CancelledError:
        print("Client disconnected. Cancelling LLM stream.")
        raise
    
    except groq.RateLimitError:
        yield ServerSentEvent(data="LLM rate limit exceeded",event="error")

    except groq.APIConnectionError:
        yield ServerSentEvent(data="Unable to connect to LLM provider",event="error")

    except groq.InternalServerError:
        yield ServerSentEvent(data="LLM provider temporarily unavailable",event="error")

    except Exception:
        yield ServerSentEvent(data="Internal streaming error",event="error")

    finally:
        if completed:
            latency = perf_counter() - start_time
            usage = client.last_usage
            if usage is not None:
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

                # Store streaming request metrics for /metrics endpoint
                metrics_store.record({
                    "model": model,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "latency": latency,
                    "cost": cost,
                })

                yield ServerSentEvent(
                    data=str(metadata),
                    event="metadata"
                )
                yield ServerSentEvent(data=str(latency),event="latency")
        
        yield ServerSentEvent(data=model, event="model")
        yield ServerSentEvent(raw_data="[DONE]",event="done")
        print("Stream cleanup complete.")