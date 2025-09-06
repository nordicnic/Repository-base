import os

from openai import OpenAI
from pydantic import BaseModel

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your API keys
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

compeltition = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an helpful assistant."},
        {
            "role": "user",
            "content": "Write a limeric about the Python programming language.",
        },
    ],
)

response = compeltition.choices[0].message.content

print(response)