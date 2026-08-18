import asyncio
import random
import traceback
from groq import AsyncGroq
import groq

from app.core.config import (
    GROQ_API_KEY,
    MAX_ATTEMPTS,
    MAX_BACKOFF,
    SEMAPHORE_TIMEOUT,
    REQUEST_TIMEOUT,
)

from app.core.exceptions import (
    NON_RETRYABLE_ERRORS,
    RETRYABLE_ERRORS,
)


class AsyncLLMClient():
    def __init__(self, model: str, semaphore: asyncio.Semaphore, circuit, temperature: float = 0.95, max_completion_tokens: int = 1024, streaming: bool = True):
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.semaphore = semaphore
        self.circuit = circuit
        self.streaming = streaming
        self.last_usage = None
        self.client = AsyncGroq(api_key=GROQ_API_KEY, max_retries=0) # we are diasbling the built-in retry mechanism to implement our own with exponential backoff and jitter

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

        await self.circuit.before_call()

        try:
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=SEMAPHORE_TIMEOUT
            )
            semaphore_acquired = True

            async with asyncio.timeout(REQUEST_TIMEOUT):

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
                        self.circuit.record_success()

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
                            self.circuit.record_failure()
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
                self.semaphore.release()

    async def generate(self, prompt: str):
        semaphore_acquired = False

        # 1. Check circuit BEFORE doing anything
        await self.circuit.before_call()

        try:
            # 2. Acquire concurrency slot
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=SEMAPHORE_TIMEOUT
            )
            semaphore_acquired = True

            async with asyncio.timeout(REQUEST_TIMEOUT):

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
                        self.circuit.record_success()

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
                            # All retries failed > circuit failure
                            self.circuit.record_failure()
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
                self.semaphore.release() 