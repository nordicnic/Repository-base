import json
import os

import requests
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your API keys
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

file_name = "wealth_evolution.xlsx"
file_path = os.path.join("/Users/nick/Library/CloudStorage/OneDrive-Personligt/Repositories/AI-Agent-Tutorial/data", file_name)

# Define a tool function
def read_expense_excel(question: str):
    """
    Reads an Excel file from the data folder and returns a Pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    df = pd.read_excel(file_path)[:100]
    return df.to_json(orient="records") 

def search_expenses(question: str):
    """
    Reads the Excel file and returns raw data.
    In a real agent, you could parse the question to filter/summarize the data.
    """
    df = read_expense_excel(question)
    # For now, just return first few rows for demo
    return df.head().to_dict(orient="records")

# 1. Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "search_expenses",
        "description": "Retrieve relevant expense data from the user's Excel file based on a natural language question.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question about their expenses (e.g., 'How much did I spend on groceries in July?')."
                }               
            },
            "required": ["question"],
            "additionalProperties": False
        },
        "strict": True
    }
]

system_prompt = "You are a helpful financial assistant that analyze the expense history of the user to provide valuable insights to answer the user's questions"

# Input conversation so far
input_list = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the largest spending category?"}
]

# 2. Call the model with tools defined
response = client.responses.create(
    model="gpt-4.1",   # or "gpt-5" if you want reasoning-heavy
    tools=tools,
    input=input_list,
)

# 3. Inspect response content
response.model_dump()
    
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
    {"answer": read_expense_excel(function_call_arguments["question"])}  
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
     