#!/usr/bin/env python3
"""
Test script for UC Tools and Vector Search functionality in simple agent
"""

import os
import logging
from agent_simple import get_simple_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tools_functionality():
    """Test UC tools and vector search functionality"""
    
    print("🛠️ Testing UC Tools and Vector Search Functionality")
    print("=" * 60)
    
    # Test without tools first
    print("\n🔍 Test 1: Simple agent without tools")
    try:
        agent = get_simple_agent()
        result = agent.predict([{"role": "user", "content": "Hello, how are you?"}])
        print(f"✅ Response: {result['response'][:100]}...")
        print(f"✅ Tools used: {result['tools_used']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test with mock UC tools (will fail without proper setup but shows loading)
    print("\n🔧 Test 2: Testing UC Tools loading (will fail without actual tools)")
    os.environ["UC_TOOL_NAMES"] = "example.catalog.function1,example.catalog.function2"
    
    try:
        from agent_simple import tools, UC_TOOL_NAMES
        print(f"✅ UC_TOOL_NAMES configured: {UC_TOOL_NAMES}")
        print(f"✅ Total tools loaded: {len(tools)}")
        
    except Exception as e:
        print(f"⚠️  Expected error (no actual UC functions): {e}")
    
    # Test with mock vector search (will fail without actual indexes)
    print("\n🔍 Test 3: Testing Vector Search Tools loading (will fail without actual indexes)")
    os.environ["VECTOR_SEARCH_INDEX_NAMES"] = "example_index1,example_index2"
    
    try:
        # Re-import to pick up new env vars
        import importlib
        import agent_simple
        importlib.reload(agent_simple)
        
        from agent_simple import VECTOR_SEARCH_TOOLS
        print(f"✅ Vector search tools configured: {len(VECTOR_SEARCH_TOOLS)}")
        
    except Exception as e:
        print(f"⚠️  Expected error (no actual vector indexes): {e}")
    
    print("\n📋 Configuration Guide:")
    print("To enable UC Tools:")
    print('  export UC_TOOL_NAMES="your_catalog.your_schema.your_function"')
    print("\nTo enable Vector Search:")
    print('  export VECTOR_SEARCH_INDEX_NAMES="your_vector_index"')
    print("\n✅ Tools functionality test completed!")

if __name__ == "__main__":
    test_tools_functionality()
