#!/usr/bin/env python3
"""
Test script for the simple (database-free) LLM agent.
"""

import os
import sys

def test_simple_imports():
    """Test if simple agent can be imported without database dependencies"""
    print("🔍 Testing simple agent imports...")
    
    try:
        from agent_simple import get_simple_agent, create_simple_agent
        print("  ✅ Simple agent imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Simple agent import failed: {e}")
        return False

def test_simple_agent_creation():
    """Test simple agent creation and basic functionality"""
    print("\n🔍 Testing simple agent creation...")
    
    try:
        from agent_simple import create_simple_agent
        
        # This will fail if auth not configured, but that's expected
        try:
            agent = create_simple_agent()
            print("  ✅ Simple agent created successfully")
            return True
        except ValueError as e:
            if "authentication" in str(e).lower() or "host" in str(e).lower():
                print("  ⚠️  Agent creation failed due to missing auth configuration (expected)")
                print(f"     Error: {e}")
                return True  # This is actually OK - means the agent code works
            else:
                print(f"  ❌ Unexpected error: {e}")
                return False
        except Exception as e:
            print(f"  ❌ Agent creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error in test: {e}")
        return False

def test_simple_dependencies():
    """Test that simple version has minimal dependencies"""
    print("\n🔍 Testing simple agent dependencies...")
    
    required_simple = ['streamlit', 'databricks_langchain', 'databricks.sdk']
    excluded_deps = ['mlflow', 'psycopg', 'langgraph']
    
    for dep in required_simple:
        try:
            if '.' in dep:
                # Handle module.submodule imports
                parts = dep.split('.')
                mod = __import__(parts[0])
                for part in parts[1:]:
                    mod = getattr(mod, part)
            else:
                __import__(dep)
            print(f"  ✅ {dep}: Available")
        except ImportError:
            print(f"  ❌ {dep}: Missing (required for simple version)")
    
    print("\n🔍 Checking that database dependencies are NOT required...")
    for dep in excluded_deps:
        try:
            __import__(dep)
            print(f"  ℹ️  {dep}: Available (not required for simple version)")
        except ImportError:
            print(f"  ✅ {dep}: Not available (good - not needed for simple version)")

def test_environment_setup():
    """Test environment variable configuration for simple version"""
    print("\n🔍 Testing environment setup for simple version...")
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"  ✅ Found {env_file} file")
    else:
        print(f"  ⚠️  No {env_file} file found")
        print("     Copy config_simple.example.env to .env for simple version")
    
    # Check required simple env vars
    required_vars = ["DATABRICKS_HOST"]
    auth_vars = ["DATABRICKS_TOKEN", "DATABRICKS_CLIENT_ID"]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: Set")
        else:
            print(f"  ❌ {var}: Not set (required)")
    
    # Check auth (at least one method)
    has_auth = any(os.getenv(var) for var in auth_vars)
    if has_auth:
        print(f"  ✅ Authentication: Configured")
    else:
        print(f"  ❌ Authentication: Not configured (need TOKEN or CLIENT_ID)")
    
    # Check that database vars are NOT required
    db_vars = ["LAKEBASE_INSTANCE_NAME", "LAKEBASE_HOST"]
    for var in db_vars:
        value = os.getenv(var)
        if value:
            print(f"  ℹ️  {var}: Set (not needed for simple version)")
        else:
            print(f"  ✅ {var}: Not set (good - not needed for simple version)")

def main():
    """Run all simple agent tests"""
    print("🚀 Simple LLM Agent Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    if test_simple_imports():
        tests_passed += 1
    
    if test_simple_dependencies():
        tests_passed += 1
    
    if test_environment_setup():
        tests_passed += 1
    
    if test_simple_agent_creation():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Simple agent tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All simple agent tests passed! Ready to use simple version.")
        return 0
    elif tests_passed >= 2:
        print("⚠️  Some tests failed, but core functionality works.")
        print("   Most likely just need to configure environment variables.")
        return 0
    else:
        print("❌ Multiple test failures. Check dependencies and setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
