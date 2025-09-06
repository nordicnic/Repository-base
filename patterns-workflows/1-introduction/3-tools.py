import json
import os

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your API keys
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Define a tool function
def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True
     }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data["current_weather"]

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Retrieves current weather for the given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"}
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }
]

system_prompt = "You are a helpful weather assistant. Always call the weather tool to get real data."

# Input conversation so far
input_list = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Copenhagen, today?"}
]

# 2. Call the model with tools defined
response = client.responses.create(
    model="gpt-4.1",   # or "gpt-5" if you want reasoning-heavy
    tools=tools,
    input=input_list,
)

# 3. Inspect response content
response.model_dump()

def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    
# 4. Extract tool calls (if any)
for item in response.output:
    data = item.model_dump()
    if data["type"] == "function_call":
        print(data["name"], data["arguments"])

# Save function call outputs for subsequent requests
function_call = None
function_call_arguments = None
input_list += response.output

for item in response.output:
    if item.type == "function_call":
        function_call = item
        function_call_arguments = json.loads(item.arguments)


# 3. Execute the function logic for get_horoscope
result = [
    {"weather": get_weather(function_call_arguments["latitude"], function_call_arguments["longitude"])}  
]

# 4. Provide function call results to the model
input_list.append({
    "type": "function_call_output",
    "call_id": function_call.call_id,
    "output": json.dumps(result),
})

print("Final input:")
print(input_list)

system_prompt = """
You are a helpful assistant. Respond in JSON format with:
{
  "temperature": float,
  "response": string
}
"""

response_2 = client.responses.create(
    model="gpt-5",
    instructions=system_prompt,
    tools=tools,
    input=input_list,
)

# 5. The model should be able to give a response!
print("Final output:")
print(json.loads(response_2.output_text)["response"])
     