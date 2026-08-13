import os
from dotenv import load_dotenv
from groq import AsyncGroq
import groq
import asyncio
import random
from time import perf_counter
from fastapi import FastAPI

app = FastAPI()

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

class AsyncLLMClient():
    def __init__(self, model: str, temperature: float = 0.95, max_completion_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
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

    async def generate(self, prompt: str):
        try:
            async with asyncio.timeout(4):
                
                for attempt in range(1,MAX_ATTEMPTS + 1):
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
                                model=self.model, temperature=self.temperature, max_completion_tokens=self.max_completion_tokens)
                        
                        return chat_completion.choices[0].message.content

                    except NON_RETRYABLE_ERRORS as e:
                        print(type(e).__name__)
                        print("Non-retryable error.")
                        raise

                    except RETRYABLE_ERRORS as e:

                        if attempt == MAX_ATTEMPTS:
                            raise
                        delay = self._get_retry_delay(e, attempt)  
                        print(f"Retryable error. Retrying in {delay:.2f}s...")
                        await asyncio.sleep(delay)

                    except Exception as e:
                        print(type(e).__name__)
                        print("Unexpected error.")
                        raise    
        except TimeoutError:
            print("Request timed out.")
            raise                    


async def main():
    client = AsyncLLMClient(model="groq/compound")

    result = await client.generate("Explain asyncio in simple terms")
    print(result)


asyncio.run(main())