import os
from dotenv import load_dotenv
from groq import AsyncGroq
import asyncio


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not set")



class AsyncLLMClient():
    def __init__(self, model: str, temperature: float = 0.95, max_completion_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    async def generate(self, prompt: str):
        client = AsyncGroq(api_key=api_key)
        
        chat_completion = await client.chat.completions.create( 
                messages=[
                    
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    # Set a user message for the assistant to respond to.
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model, temperature=self.temperature, max_completion_tokens=self.max_completion_tokens)
        
        return chat_completion.choices[0].message.content


async def main():
    client = AsyncLLMClient(model="groq/compound")
    result = await client.generate("Explain asyncio in simple terms")
    print(result)

asyncio.run(main())

