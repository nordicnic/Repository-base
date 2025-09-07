import json
import os
from typing import Dict, Any, List, Optional
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# OpenAI imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES AS NEEDED
# ============================================================================

# Database settings
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_TABLE = "expenses"

# Query settings - Edit these SQL queries as needed
DEFAULT_EXPENSES_QUERY = f"""
    SELECT * FROM {DB_TABLE} 
    LIMIT 3000
"""

# You can add more predefined queries here
CATEGORY_SUMMARY_QUERY = f"""
    SELECT category, SUM(amount) as total_amount, COUNT(*) as transaction_count
    FROM {DB_TABLE}
    GROUP BY category
    ORDER BY total_amount DESC
"""

MONTHLY_TRENDS_QUERY = f"""
    SELECT 
        DATE_TRUNC('month', date) as month,
        SUM(amount) as total_amount,
        COUNT(*) as transaction_count
    FROM {DB_TABLE}
    GROUP BY DATE_TRUNC('month', date)
    ORDER BY month DESC
    LIMIT 12
"""

# Query mapping for common questions (easily extendable)
QUERY_MAPPING = {
    # "category": CATEGORY_SUMMARY_QUERY,
    # "monthly": MONTHLY_TRENDS_QUERY,
    # "trends": MONTHLY_TRENDS_QUERY,
    "default": DEFAULT_EXPENSES_QUERY
}

# Model settings
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4.1-mini" # Be carefull with the tokens limits

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

class DatabaseConnection:
    """Handles PostgreSQL database connections and queries"""
    
    def __init__(self):
        self.connection_params = {
            'host': DB_HOST,
            'port': DB_PORT,
            'database': DB_NAME,
            'user': DB_USER,
            'password': DB_PASSWORD
        }
    
    def get_connection(self):
        """Create and return a database connection"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            return conn
        except psycopg2.Error as e:
            raise Exception(f"Database connection failed: {e}")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute query and return results as list of dictionaries"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except psycopg2.Error as e:
            raise Exception(f"Query execution failed: {e}")
        finally:
            if conn:
                conn.close()

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
        self.db = DatabaseConnection()
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

    def determine_query_type(self, question: str) -> str:
        """Determine which query to use based on the question content"""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ["category", "categories", "largest spending"]):
            return "category"
        elif any(keyword in question_lower for keyword in ["monthly", "trends", "month", "trend"]):
            return "monthly"
        else:
            return "default"

    def get_expense_data(self, question: str) -> str:
        """Retrieve expense data from PostgreSQL database"""
        try:
            query_type = self.determine_query_type(question)
            sql_query = QUERY_MAPPING.get(query_type, DEFAULT_EXPENSES_QUERY)
            
            results = self.db.execute_query(sql_query)
            return json.dumps(results, default=str, indent=2)
            
        except Exception as e:
            return f"Error retrieving data: {str(e)}"

    @tool
    def search_expenses_tool(self, question: str) -> str:
        """
        Retrieve relevant expense data from the PostgreSQL database based on a natural language question.
        
        Args:
            question: The user's question about their expenses
        """
        return self.get_expense_data(question)

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
            tool_result = self.get_expense_data(tool_call["args"]["question"])
            
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
    print("=== OpenAI via LangChain with PostgreSQL ===")
    try:
        agent = ProviderAgnosticAgent(provider=DEFAULT_PROVIDER, model=DEFAULT_MODEL)
        response = agent.query(SAMPLE_QUESTIONS[0])  # Using first sample question
        print(response)
    except Exception as e:
        print(f"Error: {e}")

def test_database_connection():
    """Test the database connection"""
    try:
        db = DatabaseConnection()
        results = db.execute_query("SELECT COUNT(*) as total_records FROM expenses")
        print(f"Database connection successful. Total records: {results[0]['total_records']}")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    # Test database connection first
    if test_database_connection():
        main()
    else:
        print("Please check your database configuration in the .env file")

# Easy agent creation function
def create_agent(provider_name: str = DEFAULT_PROVIDER, model: str = None):
    """Factory function to create agents for different providers"""
    return ProviderAgnosticAgent(provider=provider_name, model=model)

# Usage examples:
# agent = create_agent()  # Uses defaults
# response = agent.query(SAMPLE_QUESTIONS[1])  # Use any sample question
# response = agent.query("Your custom question here")