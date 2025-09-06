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
def search_kb(question: str):
    with open("kb.json", "r") as f:
        return json.load(f)

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "search_kb",
        "description": "Get the answer of the user's question from the knowledge base document.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}
            },
            "required": ["question"],
            "additionalProperties": False
        },
        "strict": True
    }
]

system_prompt = "You are a helpful assistant that answers questions"

# Input conversation so far
input_list = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"}
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
    if name == "search_kb":
        return search_kb(**args)
    
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
    {"answer": search_kb(function_call_arguments["question"])}  
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

# Check on not available answers

input_list_3 = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Rome?"}
]
    
response_3 = client.responses.create(
    model="gpt-5",
    instructions=system_prompt,
    tools=tools,
    input=input_list_3,
)

print("Final output:")
print(json.loads(response_3.output_text)["response"])
     