import json
import os
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

# Provider switching options - uncomment the one you want to use

# Option 1: LangChain approach
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic

# Option 2: LiteLLM approach (comment out LangChain imports above if using this)
import litellm
from litellm import completion

# Load environment variables
load_dotenv()

class ProviderAgnosticAgent:
    def __init__(self, provider: str = "openai", model: str = None):
        """
        Initialize agent with specified provider
        
        Args:
            provider: "openai", "ollama", "anthropic", or "litellm"
            model: specific model name (optional, uses defaults)
        """
        self.provider = provider
        self.model = model
        self.setup_client()
        
    def setup_client(self):
        """Setup the appropriate client based on provider"""
        if self.provider == "langchain_openai":
            self.client = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self.model or "gpt-4-turbo"
            )
        elif self.provider == "langchain_ollama":
            # from langchain_ollama import ChatOllama
            self.client = ChatOllama(model=self.model or "llama2")
        elif self.provider == "langchain_anthropic":
            # from langchain_anthropic import ChatAnthropic
            self.client = ChatAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model=self.model or "claude-3-sonnet-20240229"
            )
        elif self.provider == "litellm":
            # LiteLLM doesn't need a client setup
            self.model = self.model or "gpt-4-turbo"
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def read_expense_excel(self, question: str) -> str:
        """Reads an Excel file and returns JSON string"""
        file_name = "wealth_evolution.xlsx"
        file_path = os.path.join(
            "/Users/nick/Library/CloudStorage/OneDrive-Personligt/Repositories/AI-Agent-Tutorial/data", 
            file_name
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        df = pd.read_excel(file_path)[:100]
        return df.to_json(orient="records")

    # LangChain tool definition
    @tool
    def search_expenses_tool(self, question: str) -> str:
        """
        Retrieve relevant expense data from the user's Excel file based on a natural language question.
        
        Args:
            question: The user's question about their expenses
        """
        return self.read_expense_excel(question)

    def get_langchain_tools(self):
        """Return LangChain compatible tools"""
        return [self.search_expenses_tool]

    def call_with_langchain(self, messages: List[Dict], system_prompt: str) -> str:
        """Handle LangChain provider calls"""
        # Convert messages to LangChain format
        lc_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        # Bind tools to the model
        tools = self.get_langchain_tools()
        model_with_tools = self.client.bind_tools(tools)
        
        # First call to get tool usage
        response = model_with_tools.invoke(lc_messages)
        
        # If model wants to use tools, execute them
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_result = self.read_expense_excel(tool_call["args"]["question"])
            
            # Create follow-up call with tool results
            final_response = self.client.invoke([
                SystemMessage(content="You are a helpful financial assistant. Analyze the provided expense data and answer the user's question concisely."),
                HumanMessage(content=f"User question: {messages[-1]['content']}\n\nExpense data: {tool_result}")
            ])
            return final_response.content
        
        return response.content

    def call_with_litellm(self, messages: List[Dict], system_prompt: str) -> str:
        """Handle LiteLLM provider calls"""
        # Prepare messages
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Define tools in OpenAI format (LiteLLM uses OpenAI-compatible format)
        tools = [{
            "type": "function",
            "function": {
                "name": "search_expenses",
                "description": "Retrieve expense data from Excel file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "User's question about expenses"
                        }
                    },
                    "required": ["question"]
                }
            }
        }]
        
        # First call
        response = completion(
            model=self.model,
            messages=full_messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Check if model wants to use tools
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_call = message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            tool_result = self.read_expense_excel(function_args["question"])
            
            # Second call with tool results
            final_messages = full_messages + [
                message.dict(),
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                }
            ]
            
            final_response = completion(
                model=self.model,
                messages=final_messages
            )
            return final_response.choices[0].message.content
        
        return message.content

    def query(self, user_question: str, system_prompt: str = None) -> str:
        """
        Main query method that works across providers
        
        Args:
            user_question: User's question
            system_prompt: System prompt (optional)
            
        Returns:
            Agent's response
        """
        if system_prompt is None:
            system_prompt = "You are a helpful financial assistant that analyzes expense history to provide valuable insights."
        
        messages = [{"role": "user", "content": user_question}]
        
        if self.provider.startswith("langchain"):
            return self.call_with_langchain(messages, system_prompt)
        elif self.provider == "litellm":
            return self.call_with_litellm(messages, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

# Usage examples:

def main():
    # Example 1: Using OpenAI via LangChain
    print("=== OpenAI via LangChain ===")
    agent_openai = ProviderAgnosticAgent(provider="langchain_openai", model="gpt-4-turbo")
    response = agent_openai.query("What is the largest spending category?")
    print(response)
    
    # Example 2: Using LiteLLM (can switch between providers easily)
    print("\n=== OpenAI via LiteLLM ===")
    agent_litellm = ProviderAgnosticAgent(provider="litellm", model="gpt-4-turbo")
    response = agent_litellm.query("What is the largest spending category?")
    print(response)
    
    # Example 3: Switch to Ollama via LiteLLM (uncomment if you have Ollama running)
    # print("\n=== Ollama via LiteLLM ===")
    # agent_ollama = ProviderAgnosticAgent(provider="litellm", model="ollama/llama2")
    # response = agent_ollama.query("What is the largest spending category?")
    # print(response)
    
    # Example 4: Test with unavailable tool scenario
    print("\n=== Testing unavailable tool scenario ===")
    response = agent_openai.query("What's the weather like in Rome?")
    print(response)

if __name__ == "__main__":
    main()

# Configuration examples for different providers:

# For LiteLLM provider switching:
LITELLM_MODELS = {
    "openai": "gpt-4-turbo",
    "anthropic": "claude-3-sonnet-20240229", 
    "ollama": "ollama/llama2",
    "cohere": "command-r-plus",
    "gemini": "gemini-pro"
}

# Easy provider switching function
def create_agent(provider_name: str, db_config: Dict = None):
    """Factory function to create agents for different providers"""
    if provider_name in LITELLM_MODELS:
        return ProviderAgnosticAgent(
            provider="litellm", 
            model=LITELLM_MODELS[provider_name],
            db_config=db_config
        )
    elif provider_name == "langchain_openai":
        return ProviderAgnosticAgent(provider="langchain_openai", db_config=db_config)
    elif provider_name == "langchain_ollama":
        return ProviderAgnosticAgent(provider="langchain_ollama", db_config=db_config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

# Usage:
# db_config = {'host': 'localhost', 'database': 'expenses', 'user': 'postgres', 'password': 'your_password'}
# agent = create_agent("anthropic", db_config)  # Switch to Claude
# agent = create_agent("ollama", db_config)     # Switch to local Ollama
# response = agent.query("Your question here")