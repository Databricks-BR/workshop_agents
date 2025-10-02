#!/usr/bin/env python3
"""
Test environment variable configuration without heavy dependencies
"""

import os
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_env_vars():
    print("🔍 Testing environment variable configuration...")
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ Found {env_file} file")
        try:
            # Try to load .env if python-dotenv is available
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Loaded environment variables from .env")
        except ImportError:
            print("⚠️  python-dotenv not available, using system environment variables only")
            print("   Tip: pip install python-dotenv to load .env files automatically")
    else:
        print(f"⚠️  No {env_file} file found")
        print("   Copy config.example.env to .env and configure your values")
    
    print("\n📋 Current environment variables:")
    
    # Test the same logic as agent.py
    env_vars = {
        "DATABRICKS_HOST": os.getenv("DATABRICKS_HOST"),
        "DATABRICKS_CLIENT_ID": os.getenv("DATABRICKS_CLIENT_ID"),
        "DATABRICKS_CLIENT_SECRET": os.getenv("DATABRICKS_CLIENT_SECRET"),
        "DATABRICKS_TOKEN": os.getenv("DATABRICKS_TOKEN"),
        "LLM_ENDPOINT_NAME": os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-3-7-sonnet"),
        "LAKEBASE_INSTANCE_NAME": os.getenv("LAKEBASE_INSTANCE_NAME"),
        "LAKEBASE_HOST": os.getenv("LAKEBASE_HOST"),
        "LAKEBASE_DB_NAME": os.getenv("LAKEBASE_DB_NAME", "databricks_postgres"),
        "LAKEBASE_SSL_MODE": os.getenv("LAKEBASE_SSL_MODE", "require"),
    }
    
    for key, value in env_vars.items():
        if value:
            if "SECRET" in key or "TOKEN" in key:
                print(f"  ✅ {key}: ***{value[-4:] if len(value) >= 4 else '***'}")
            else:
                print(f"  ✅ {key}: {value}")
        else:
            print(f"  ❌ {key}: Not set")
    
    # Test validation logic
    print("\n🧪 Testing validation logic:")
    
    missing_vars = []
    if not env_vars["LAKEBASE_INSTANCE_NAME"]:
        missing_vars.append("LAKEBASE_INSTANCE_NAME")
    if not env_vars["LAKEBASE_HOST"]:
        missing_vars.append("LAKEBASE_HOST")
    
    if missing_vars:
        print(f"❌ Missing required Lakebase environment variables: {', '.join(missing_vars)}")
        print("Please set these environment variables:")
        for var in missing_vars:
            print(f"  export {var}=your-value-here")
        print("\nOr create a .env file with:")
        for var in missing_vars:
            print(f"  {var}=your-value-here")
        return False
    else:
        print("✅ All required Lakebase environment variables are set!")
        return True

if __name__ == "__main__":
    success = test_env_vars()
    if success:
        print("\n🎉 Environment configuration looks good!")
        exit(0)
    else:
        print("\n⚠️  Please fix the missing environment variables above")
        exit(1)
