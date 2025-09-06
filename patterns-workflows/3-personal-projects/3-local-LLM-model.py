import json
import os
import requests
import pandas as pd
from typing import Dict, List, Any

class LocalExpenseAnalyzer:
    """Expense analyzer optimized for local Ollama on M3 MacBook"""
    
    def __init__(self, model_name: str = "mistral:7b-instruct-q4_K_M", 
                 ollama_url: str = "http://localhost:11434"):
        self.model = model_name
        self.base_url = ollama_url
        self.file_path = "/Users/nick/Library/CloudStorage/OneDrive-Personligt/Repositories/AI-Agent-Tutorial/data/wealth_evolution.xlsx"
    
    def read_expense_excel(self, question: str = None) -> str:
        """Read Excel file and return JSON data"""
        try:
            if not os.path.exists(self.file_path):
                return json.dumps({"error": f"File not found: {self.file_path}"})
            
            df = pd.read_excel(self.file_path)[:100]
            return df.to_json(orient="records")
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def analyze_expenses_data(self, data: str, question: str) -> Dict:
        """Analyze expense data using local LLM"""
        prompt = f"""
You are a financial analyst. Analyze the following expense data and answer the question.

Expense Data (JSON format):
{data[:2000]}...  

Question: {question}

Provide a clear, concise analysis focusing on:
1. Direct answer to the question
2. Key insights from the data
3. Relevant spending patterns

Response:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower temperature for more consistent analysis
                        "top_p": 0.9,
                        "num_ctx": 4096     # Context window
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return {
                "success": True,
                "response": response.json()["response"],
                "model_used": self.model
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Ollama request failed: {str(e)}",
                "fallback": "Please ensure Ollama is running with: ollama serve"
            }
    
    def chat_about_expenses(self, question: str) -> str:
        """Main interface - ask any question about expenses"""
        
        # Check if question is about expenses
        expense_keywords = ["spend", "expense", "cost", "money", "budget", "category", "total", "amount"]
        is_expense_question = any(keyword in question.lower() for keyword in expense_keywords)
        
        if is_expense_question:
            # Read expense data
            expense_data = self.read_expense_excel()
            
            if "error" in expense_data:
                return f"Error reading expense file: {json.loads(expense_data)['error']}"
            
            # Analyze with LLM
            analysis = self.analyze_expenses_data(expense_data, question)
            
            if analysis["success"]:
                return analysis["response"]
            else:
                return f"Analysis failed: {analysis['error']}"
        else:
            # Handle non-expense questions
            return self.general_chat(question)
    
    def general_chat(self, question: str) -> str:
        """Handle general questions without expense data"""
        prompt = f"""You are a helpful assistant. The user asked: {question}

If this question is about expenses or financial data, explain that you need access to expense data to answer properly.
Otherwise, provide a helpful response.

Response:"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except:
            return "Sorry, I couldn't process your question. Please ensure Ollama is running."
    
    def check_ollama_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and which models are available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = response.json()["models"]
            
            # Check if our model is available
            model_available = any(self.model in model["name"] for model in models)
            
            return {
                "ollama_running": True,
                "model_available": model_available,
                "available_models": [model["name"] for model in models],
                "recommended_action": "Ready to use!" if model_available else f"Run: ollama pull {self.model}"
            }
        except:
            return {
                "ollama_running": False,
                "model_available": False,
                "available_models": [],
                "recommended_action": "Start Ollama with: ollama serve"
            }

# Easy-to-use wrapper
def quick_expense_chat(question: str, model: str = "mistral:7b-instruct-q4_K_M") -> str:
    """One-function interface for quick questions"""
    analyzer = LocalExpenseAnalyzer(model)
    status = analyzer.check_ollama_status()
    
    if not status["ollama_running"]:
        return "❌ Ollama is not running. Start it with: ollama serve"
    
    if not status["model_available"]:
        return f"❌ Model {model} not found. Install with: ollama pull {model}"
    
    return analyzer.chat_about_expenses(question)

# Example usage
if __name__ == "__main__":
    # Method 1: Direct usage
    analyzer = LocalExpenseAnalyzer()
    
    # Check system status
    status = analyzer.check_ollama_status()
    print("System Status:", status)
    
    if status["ollama_running"] and status["model_available"]:
        # Test questions
        questions = [
            "What is the largest spending category?",
            "How much did I spend in total?",
            "What's the weather like in Rome?"  # Non-expense question
        ]
        
        for q in questions:
            print(f"\n🔍 Question: {q}")
            answer = analyzer.chat_about_expenses(q)
            print(f"💡 Answer: {answer}")
    
    # Method 2: Quick function (one-liner)
    print("\n" + "="*50)
    print("Quick Demo:")
    result = quick_expense_chat("What is my biggest expense?")
    print(result)

# Benchmarking function for M3 performance
def benchmark_local_models():
    """Test different models on M3 MacBook Air"""
    models_to_test = [
        "mistral:7b-instruct-q4_K_M",  # Lightest, fastest
        "llama2:7b-chat",              # Most compatible
        "deepseek-coder:6.7b-instruct-q4_0",  # Best for code
        "phi3:mini"                    # Microsoft's efficient model
    ]
    
    test_question = "Analyze my top 3 spending categories"
    
    for model in models_to_test:
        print(f"\n🧪 Testing {model}...")
        try:
            import time
            start_time = time.time()
            
            analyzer = LocalExpenseAnalyzer(model)
            status = analyzer.check_ollama_status()
            
            if status["model_available"]:
                result = analyzer.chat_about_expenses(test_question)
                elapsed = time.time() - start_time
                print(f"✅ Response time: {elapsed:.2f}s")
                print(f"📊 Response preview: {result[:100]}...")
            else:
                print(f"❌ Model not installed. Run: ollama pull {model}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

# Run benchmark if needed
# benchmark_local_models()