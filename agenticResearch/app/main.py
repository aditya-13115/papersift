import os
import traceback
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from groq import AsyncGroq
import groq
import asyncio
import random
import time
from time import perf_counter
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.sse import EventSourceResponse, ServerSentEvent
from collections.abc import AsyncIterable
from model_router import SemanticRouter
from metrics import MetricsStore


app = FastAPI()
metrics_store = MetricsStore(max_size=1000)
semaphore = asyncio.Semaphore(10)
router = SemanticRouter()


PRICING = { # pricing are $ per 1M tokens.
    "openai/gpt-oss-20b": {
        "input": 0.075,
        "output": 0.30
    },

    "openai/gpt-oss-120b": {
        "input": 0.15,
        "output": 0.60
    }
}


class GenerateRequest(BaseModel):
    prompt: str

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set")

MAX_ATTEMPTS = 3
MAX_BACKOFF = 30 #seconds

NON_RETRYABLE_ERRORS = (
    groq.BadRequestError,          # 400
    groq.AuthenticationError,      # 401
    groq.PermissionDeniedError,    # 403
    groq.NotFoundError,            # 404
    groq.UnprocessableEntityError, # 422
)

RETRYABLE_ERRORS = (
    groq.RateLimitError,      # 429
    groq.InternalServerError, # 500, 502, 503
    groq.APIConnectionError,
    groq.APITimeoutError
)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    async def before_call(self):
        if self.state == "OPEN":

            elapsed = time.monotonic() - self.last_failure_time

            if elapsed < self.recovery_timeout:
                raise RuntimeError("Circuit is OPEN")

            # Recovery timeout passed
            self.state = "HALF_OPEN"

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"



circuit = CircuitBreaker(failure_threshold=5,recovery_timeout=30)

class AsyncLLMClient():
    def __init__(self, model: str, temperature: float = 0.95, max_completion_tokens: int = 1024, streaming: bool = True):
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.streaming = streaming
        self.last_usage = None
        self.client = AsyncGroq(api_key=api_key,max_retries=0) # we are diasbling the built-in retry mechanism to implement our own with exponential backoff and jitter

    def _calculate_backoff(self, attempt: int) -> float:
        base_delay = min(2 ** (attempt - 1), MAX_BACKOFF)
        jitter = random.uniform(0, base_delay)
        return base_delay + jitter
    
    def _get_retry_delay(self, error, attempt: int) -> float:
        # 429 → use server-provided Retry-After if available
        if isinstance(error, groq.RateLimitError):
            retry_after = error.response.headers.get("retry-after")
            if retry_after is not None:
                return float(retry_after)
        # All other retryable errors, or 429 without Retry-After
        return self._calculate_backoff(attempt)

    async def stream(self, prompt: str):
        self.last_usage = None # reset last_usage before every stream
        semaphore_acquired = False

        await circuit.before_call()

        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=2
            )
            semaphore_acquired = True

            async with asyncio.timeout(10):

                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        stream = await self.client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant."
                                },
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            model=self.model,
                            temperature=self.temperature,
                            max_completion_tokens=self.max_completion_tokens,
                            stream=self.streaming,
                        )

                        async for chunk in stream:
                            if chunk.choices:
                                content = chunk.choices[0].delta.content

                                if content:
                                    yield content

                            if chunk.usage is not None:
                                self.last_usage = chunk.usage

                        # Entire stream completed successfully
                        circuit.record_success()

                        return

                    except asyncio.CancelledError:
                        print("LLM stream cancelled.")
                        raise

                    except NON_RETRYABLE_ERRORS as e:
                        print(type(e).__name__)
                        print("Non-retryable error.")
                        raise

                    except RETRYABLE_ERRORS as e:

                        if attempt == MAX_ATTEMPTS:
                            # All retries failed
                            circuit.record_failure()
                            raise

                        delay = self._get_retry_delay(e, attempt)

                        print(
                            f"Retryable error. "
                            f"Retrying in {delay:.2f}s..."
                        )

                        await asyncio.sleep(delay)

                    except Exception as e:
                        print(type(e).__name__)
                        print(f"Unexpected error: {e}")
                        traceback.print_exc()
                        raise

        except asyncio.TimeoutError:
            print("Semaphore wait timed out.")
            raise

        except TimeoutError:
            print("Request timed out.")
            raise

        finally:
            if semaphore_acquired:
                semaphore.release()

    async def generate(self, prompt: str):
        semaphore_acquired = False

        # 1. Check circuit BEFORE doing anything
        await circuit.before_call()

        try:
            # 2. Acquire concurrency slot
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=2
            )
            semaphore_acquired = True

            async with asyncio.timeout(10):

                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        chat_completion = await self.client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful assistant."
                                },
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            model=self.model,
                            temperature=self.temperature,
                            max_completion_tokens=self.max_completion_tokens,
                            stream=False
                        )

                        # SUCCESS
                        circuit.record_success()

                        return {
                        "text": chat_completion.choices[0].message.content,
                        "usage": chat_completion.usage
                    }

                    except NON_RETRYABLE_ERRORS as e:
                        print(type(e).__name__)
                        print("Non-retryable error.")
                        raise

                    except RETRYABLE_ERRORS as e:

                        if attempt == MAX_ATTEMPTS:
                            # All retries failed → circuit failure
                            circuit.record_failure()
                            raise

                        delay = self._get_retry_delay(e, attempt)

                        print(
                            f"Retryable error. "
                            f"Retrying in {delay:.2f}s..."
                        )

                        await asyncio.sleep(delay)

                    except Exception as e:
                        print(type(e).__name__)
                        print(f"Unexpected error: {e}")
                        traceback.print_exc()
                        raise

        except asyncio.TimeoutError:
            print("Semaphore wait timed out.")
            raise

        except TimeoutError:
            print("Request timed out.")
            raise

        finally:
            if semaphore_acquired:
                semaphore.release()                   


@app.exception_handler(TimeoutError)
async def timeout_exception_handler(request: Request, exc: TimeoutError):
    return JSONResponse(
        status_code=504,
        content={"detail": "LLM request timed out"}
    )


@app.exception_handler(groq.BadRequestError)
async def bad_request_handler(request: Request, exc: groq.BadRequestError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid LLM request"}
    )


@app.exception_handler(groq.AuthenticationError)
async def authentication_handler(request: Request, exc: groq.AuthenticationError):
    return JSONResponse(
        status_code=401,
        content={"detail": "LLM authentication failed"}
    )


@app.exception_handler(groq.PermissionDeniedError)
async def permission_handler(request: Request, exc: groq.PermissionDeniedError):
    return JSONResponse(
        status_code=403,
        content={"detail": "LLM request forbidden"}
    )


@app.exception_handler(groq.NotFoundError)
async def not_found_handler(request: Request, exc: groq.NotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": "LLM model not found"}
    )


@app.exception_handler(groq.RateLimitError)
async def rate_limit_handler(request: Request, exc: groq.RateLimitError):
    return JSONResponse(
        status_code=429,
        content={"detail": "LLM rate limit exceeded"}
    )


@app.exception_handler(groq.InternalServerError)
async def server_error_handler(request: Request, exc: groq.InternalServerError):
    return JSONResponse(
        status_code=502,
        content={"detail": "LLM provider temporarily unavailable"}
    )

@app.exception_handler(groq.APIConnectionError)
async def connection_error_handler(request: Request, exc: groq.APIConnectionError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Unable to connect to LLM provider"}
    )

@app.exception_handler(groq.APITimeoutError)
async def api_timeout_handler(request: Request, exc: groq.APITimeoutError):
    return JSONResponse(
        status_code=504,
        content={"detail": "LLM provider request timed out"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    start_time = perf_counter()
    model = router.route(request.prompt)
    print(f"ROUTED MODEL: {model}")
    client = AsyncLLMClient(model=model, max_completion_tokens=256)  # You can adjust the model and parameters as needed
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


@app.post("/stream", response_class=EventSourceResponse)
async def stream_endpoint(request: GenerateRequest) -> AsyncIterable[ServerSentEvent]:
    start_time = perf_counter()
    model = router.route(request.prompt)
    print(f"ROUTED MODEL: {model}")
    client = AsyncLLMClient(model=model, streaming=True, max_completion_tokens=256)  # You can adjust the model and parameters as needed
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


@app.get("/metrics")
async def metrics():
    return metrics_store.summary()

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }