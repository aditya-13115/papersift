import os
import json

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


# -----------------------------
# 1. Actual Python Tool
# -----------------------------

def get_weather(city: str) -> int | str:
    mock_temperatures = {
        "New York": 75,
        "Los Angeles": 85,
        "Chicago": 70,
        "Houston": 90,
        "Phoenix": 100,
    }

    return mock_temperatures.get(city, "City not found")


# -----------------------------
# 2. Tool Registry
# -----------------------------

tools = {
    "get_weather": {
        "function": get_weather,
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get the weather for.",
                }
            },
            "required": ["city"],
        },
    }
}


# -----------------------------
# 3. Convert Registry → LLM Tools
# -----------------------------

llm_tools = []

for name, tool in tools.items():
    llm_tools.append({
        "type": "function",
        "function": {
            "name": name,
            "description": tool["description"],
            "parameters": tool["parameters"],
        },
    })


# -----------------------------
# 4. Ask the LLM
# -----------------------------

prompt = "What is the weather in Chicago?"

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    tools=llm_tools,
)


message = response.choices[0].message

print(message)
# -----------------------------
# 5. Check whether LLM wants a tool
# -----------------------------

if message.tool_calls:

    tool_call = message.tool_calls[0]

    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    print("LLM selected tool:", tool_name)
    print("Arguments:", tool_args)

    # -----------------------------
    # 6. Execute the selected tool
    # -----------------------------

    tool = tools[tool_name]
    print("tool :", tool)
    print("tool arg :", tool_args)

    result = tool["function"](**tool_args)

    print("Tool result:", result)

else:
    print("LLM response:", message.content)


messages = [
    {
        "role": "user",
        "content": prompt,
    },
    message,
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
    },
]

final_response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=llm_tools,
)

print(final_response.choices[0].message.content)