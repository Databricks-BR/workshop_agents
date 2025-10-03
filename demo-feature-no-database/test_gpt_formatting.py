#!/usr/bin/env python3
"""
Test script for GPT response formatting
"""

import logging
from agent_simple import get_simple_agent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_gpt_formatting():
    """Test GPT response formatting with various message types"""
    
    print("🧪 Testing GPT Response Formatting")
    print("=" * 50)
    
    try:
        agent = get_simple_agent()
        print("✅ Agent loaded successfully")
        
        test_messages = [
            "Hello! How are you?",
            "Can you write a simple Python function with proper formatting?",
            "List 3 benefits of using databases:\n1. First benefit\n2. Second benefit\n3. Third benefit",
            "What is machine learning? Please format your answer with headers and bullet points."
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n📝 Test {i}: {message[:50]}...")
            
            try:
                result = agent.predict([{"role": "user", "content": message}])
                response = result["response"]
                
                print(f"✅ Response length: {len(response)} characters")
                print(f"✅ Line breaks in response: {response.count(chr(10))} newlines")
                print("📄 Response preview:")
                print("-" * 40)
                print(response[:300] + "..." if len(response) > 300 else response)
                print("-" * 40)
                
            except Exception as e:
                print(f"❌ Error in test {i}: {e}")
                
    except Exception as e:
        print(f"❌ Error loading agent: {e}")

if __name__ == "__main__":
    test_gpt_formatting()
