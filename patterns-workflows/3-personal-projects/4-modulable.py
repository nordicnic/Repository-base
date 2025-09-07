import json
import os
from typing import Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

# OpenAI imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES AS NEEDED
# ============================================================================

# File settings
EXCEL_FILE_NAME = "wealth_evolution.xlsx"
DATA_FOLDER_PATH = "/Users/nick/Library/CloudStorage/OneDrive-Personligt/Repositories/AI-Agent-Tutorial/data"
MAX_ROWS_TO_READ = 100

# Model settings
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4-turbo"

# Prompts
DEFAULT_SYSTEM_PROMPT = "You are a helpful financial assistant that analyzes expense history to provide valuable insights."
TOOL_ANALYSIS_PROMPT = "You are a helpful financial assistant. Analyze the provided expense data and answer the user's question concisely."

# Sample questions for testing
SAMPLE_QUESTIONS = [
    "What is the largest spending category?",
    "Show me my monthly spending trends",
    "What are my top 5 expenses this year?"
]

# ============================================================================

class ProviderAgnosticAgent:
    def __init__(self, provider: str = DEFAULT_PROVIDER, model: str = None):
        """
        Initialize agent with specified provider
        
        Args:
            provider: Currently supports DEFAULT_PROVIDER (extensible for future providers)
            model: specific model name (optional, uses defaults)
        """
        self.provider = provider
        self.model = model
        self.setup_client()
        
    def setup_client(self):
        """Setup the appropriate client based on provider"""
        if self.provider == DEFAULT_PROVIDER:
            self.client = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self.model or DEFAULT_MODEL
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Currently only 'openai' is supported.")

    def read_expense_excel(self, question: str) -> str:
        """Reads an Excel file and returns JSON string"""
        file_path = os.path.join(
            DATA_FOLDER_PATH, 
            EXCEL_FILE_NAME
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        df = pd.read_excel(file_path)[:MAX_ROWS_TO_READ]
        return df.to_json(orient="records")

    @tool
    def search_expenses_tool(self, question: str) -> str:
        """
        Retrieve relevant expense data from the user's Excel file based on a natural language question.
        
        Args:
            question: The user's question about their expenses
        """
        return self.read_expense_excel(question)

    def get_tools(self):
        """Return provider-compatible tools"""
        return [self.search_expenses_tool]

    def call_openai(self, messages: List[Dict], system_prompt: str) -> str:
        """Handle OpenAI provider calls using LangChain"""
        # Convert messages to LangChain format
        lc_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
        
        # Bind tools to the model
        tools = self.get_tools()
        model_with_tools = self.client.bind_tools(tools)
        
        # First call to get tool usage
        response = model_with_tools.invoke(lc_messages)
        
        # If model wants to use tools, execute them
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_result = self.read_expense_excel(tool_call["args"]["question"])
            
            # Create follow-up call with tool results
            final_response = self.client.invoke([
                SystemMessage(content=TOOL_ANALYSIS_PROMPT),
                HumanMessage(content=f"User question: {messages[-1]['content']}\n\nExpense data: {tool_result}")
            ])
            return final_response.content
        
        return response.content

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
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        messages = [{"role": "user", "content": user_question}]
        
        if self.provider == DEFAULT_PROVIDER:
            return self.call_openai(messages, system_prompt)
        else:
            # Future providers can be added here
            # elif self.provider == "anthropic":
            #     return self.call_anthropic(messages, system_prompt)
            raise ValueError(f"Unsupported provider: {self.provider}")

# Usage examples:
def main():
    # Using OpenAI
    print("=== OpenAI via LangChain ===")
    agent = ProviderAgnosticAgent(provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)
    response = agent.query(SAMPLE_QUESTIONS[0])  # Using first sample question
    print(response)

if __name__ == "__main__":
    main()

# Easy agent creation function
def create_agent(provider_name: str = DEFAULT_PROVIDER, model: str = None):
    """Factory function to create agents for different providers"""
    return ProviderAgnosticAgent(provider=provider_name, model=model)

# Usage examples:
# agent = create_agent()  # Uses defaults
# response = agent.query(SAMPLE_QUESTIONS[1])  # Use any sample question
# response = agent.query("Your custom question here")