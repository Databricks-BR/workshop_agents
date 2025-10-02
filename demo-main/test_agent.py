#!/usr/bin/env python3
"""
Test script for the LLM agent initialization.

This script demonstrates proper usage and troubleshooting for the agent.
Run this after installing requirements and configuring environment variables.
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    required_packages = [
        'mlflow', 'databricks_langchain', 'langgraph', 
        'psycopg', 'databricks.sdk', 'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('.', '/') if '.' in package else package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All packages available!")
    return True

def test_environment():
    """Test if required environment variables are set"""
    print("\n🔍 Testing environment variables...")
    
    required_env_vars = [
        'DATABRICKS_HOST',
        'DATABRICKS_CLIENT_ID', 
        'DATABRICKS_CLIENT_SECRET'
    ]
    
    optional_env_vars = [
        'LLM_ENDPOINT_NAME',
        'LAKEBASE_INSTANCE_NAME',
        'LAKEBASE_HOST'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: {'*' * 10}")  # Hide actual values
        else:
            print(f"  ❌ {var}: Not set")
            missing_vars.append(var)
    
    for var in optional_env_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: {'*' * 10}")
        else:
            print(f"  ⚠️  {var}: Using default")
    
    if missing_vars:
        print(f"\n⚠️  Missing required variables: {', '.join(missing_vars)}")
        print("Copy config.example.env to .env and configure your settings")
        return False
    
    print("✅ Environment variables configured!")
    return True

def test_agent_creation():
    """Test agent creation"""
    print("\n🔍 Testing agent creation...")
    
    try:
        from agent import get_agent
        print("  ✅ Agent module imported successfully")
        
        # Try to create the agent
        agent = get_agent()
        print("  ✅ Agent created successfully!")
        
        # Test a simple prediction
        test_request = {
            "input": [{"role": "user", "content": "Hello, this is a test"}]
        }
        
        print("  🔍 Testing prediction...")
        response = agent.predict(test_request)
        print(f"  ✅ Prediction successful! Thread ID: {response.custom_outputs.get('thread_id', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent creation failed: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Verify Databricks workspace access")
        print("  2. Check Lakebase instance availability") 
        print("  3. Ensure model endpoint permissions")
        print("  4. Review environment variable values")
        return False

def main():
    """Run all tests"""
    print("🚀 LLM Agent Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    
    if test_imports():
        tests_passed += 1
    
    if test_environment():
        tests_passed += 1
    
    if test_imports() and test_environment():
        if test_agent_creation():
            tests_passed += 1
    else:
        print("\n⏭️  Skipping agent creation test due to missing dependencies/config")
    
    print("\n" + "=" * 50)
    print(f"📊 Tests passed: {tests_passed}/3")
    
    if tests_passed == 3:
        print("🎉 All tests passed! Your agent is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
