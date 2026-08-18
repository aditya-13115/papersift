import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

MAX_ATTEMPTS = 3
MAX_BACKOFF = 30 #seconds
SEMAPHORE_LIMIT = 10
SEMAPHORE_TIMEOUT = 2
REQUEST_TIMEOUT = 10

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