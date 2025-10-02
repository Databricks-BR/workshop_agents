# Streamlit LLM Agent Chatbot

This Streamlit application provides a chatbot interface for interacting with a stateful LLM agent built using LangGraph and Databricks. It combines the LLM functionality from `stateful-agent-lakebase.ipynb` with a Streamlit UI similar to `stlit_main.py`.

## Features

- **Stateful Conversations**: Maintains conversation history using thread-based memory persistence in Lakebase
- **LangGraph Agent**: Uses the LangGraphResponsesAgent for intelligent responses
- **Thread Management**: Each conversation session maintains its state, allowing for context-aware responses
- **Modern UI**: Clean Streamlit interface with chat history, examples, and controls

## Prerequisites

1. **Lakebase Instance**: Make sure you have a configured Lakebase instance (same as required for the notebook)
2. **Databricks Setup**: Proper Databricks workspace configuration
3. **Agent Dependencies**: The `agent.py` file from the notebook should be in the same directory

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables
Copy `config.example.env` to `.env` and configure your settings:

```bash
cp config.example.env .env
```

Required environment variables:
- `DATABRICKS_HOST`: Your Databricks workspace URL
- `DATABRICKS_CLIENT_ID`: Service principal client ID (recommended) or user credentials
- `DATABRICKS_CLIENT_SECRET`: Service principal secret
- `LAKEBASE_INSTANCE_NAME`: Your Lakebase instance name (REQUIRED)
- `LAKEBASE_HOST`: Your Lakebase connection host (REQUIRED)

Optional environment variables:
- `DATABRICKS_TOKEN`: Personal access token (alternative to client ID/secret)
- `LLM_ENDPOINT_NAME`: Your model serving endpoint (default: "databricks-claude-3-7-sonnet")
- `LAKEBASE_DB_NAME`: Database name (default: "databricks_postgres")
- `LAKEBASE_SSL_MODE`: SSL mode (default: "require")

### Alternative: Modify agent.py directly
You can also modify the default values in `agent.py` for:
- `LLM_ENDPOINT_NAME`: Your Databricks model serving endpoint
- `LAKEBASE_CONFIG`: Your Lakebase instance configuration

## Running the Application

```bash
streamlit run streamlit_llm_agent.py
```

## Usage

1. **Start Chatting**: Type your message in the chat input at the bottom
2. **Use Examples**: Click on example prompts in the sidebar
3. **Thread Management**: 
   - The app automatically maintains conversation state using thread IDs
   - Click "Start New Thread" to begin a fresh conversation
   - Click "Clear Chat" to clear the UI (but thread state persists in Lakebase)
4. **Monitor Threads**: The current thread ID is shown in the sidebar

## Key Differences from Genie Version

- **LLM Agent**: Uses the stateful LLM agent instead of Databricks Genie API
- **Thread-based Memory**: Leverages LangGraph's checkpointing for conversation persistence
- **Response Format**: Processes LLM agent responses instead of SQL query results
- **State Management**: Uses Lakebase PostgreSQL for durable state storage

## Architecture

```
User Input → Streamlit UI → LangGraph Agent → Claude 3.5 Sonnet → Response
                              ↓
                         Lakebase (State Storage)
```

## Troubleshooting

1. **Import Errors**: Make sure `agent.py` is in the same directory and all dependencies are installed
2. **Database Connection**: Verify your Lakebase configuration and credentials
3. **Model Endpoint**: Ensure your Databricks model serving endpoint is accessible
4. **Thread Issues**: If threads aren't persisting, check your PostgreSQL connection to Lakebase
