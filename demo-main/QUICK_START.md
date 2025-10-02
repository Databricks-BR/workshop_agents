# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
# Copy the example configuration
cp config.example.env .env

# Edit .env with your values
# You need to set at least these required variables:
# - LAKEBASE_INSTANCE_NAME=your-instance-name  
# - LAKEBASE_HOST=your-host.database.cloud.databricks.com
# - DATABRICKS_TOKEN=your-token (or CLIENT_ID/CLIENT_SECRET)
```

### 3. Test Your Setup
```bash
# Test environment configuration
python test_env_vars.py

# Test agent initialization (requires all dependencies)
python test_agent.py

# Run the Streamlit app
streamlit run streamlit_llm_agent.py
```

## 🔧 Configuration Details

### Required Environment Variables
- `LAKEBASE_INSTANCE_NAME`: Your Lakebase instance name
- `LAKEBASE_HOST`: Your Lakebase host URL
- `DATABRICKS_TOKEN`: Your Databricks personal access token

### Optional Environment Variables  
- `LLM_ENDPOINT_NAME`: Model endpoint (default: databricks-claude-3-7-sonnet)
- `LAKEBASE_DB_NAME`: Database name (default: databricks_postgres)
- `LAKEBASE_SSL_MODE`: SSL mode (default: require)

## 🐛 Troubleshooting

### Environment Variables Not Loading
- Make sure `.env` file exists: `ls -la .env`
- Check file contents: `cat .env`
- Install dotenv: `pip install python-dotenv`

### Missing Dependencies
- Install all: `pip install -r requirements.txt`
- Test imports: `python -c "import mlflow, databricks_langchain"`

### Agent Creation Fails
- Test environment: `python test_env_vars.py`
- Check Databricks access: verify your token/credentials
- Verify Lakebase instance is accessible

### Import Errors
The agent now uses lazy initialization - it won't fail on import, only when you try to create the agent instance. This allows you to:
- Install dependencies gradually
- Configure environment variables after import
- Get better error messages when something is missing
