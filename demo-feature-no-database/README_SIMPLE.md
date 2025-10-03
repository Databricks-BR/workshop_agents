# Simple LLM Agent - Database-Free Version

This is a simplified version of the LLM agent that **does not require any database setup**. It uses in-memory conversation state management instead of Lakebase/PostgreSQL persistence.

## 🎯 Key Differences from Main Version

| Feature | Main Version | Simple Version |
|---------|--------------|----------------|
| **State Storage** | Lakebase PostgreSQL | In-memory (Streamlit session) |
| **Conversation Persistence** | Permanent (survives restarts) | Session-only (lost on restart) |
| **Database Setup** | Required (Lakebase instance) | None required |
| **Dependencies** | ~10 packages including PostgreSQL | 5 lightweight packages |
| **Setup Complexity** | High (database, credentials) | Low (just Databricks auth) |

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_simple.txt
```

### 2. Configure Environment
```bash
# Copy the simple configuration
cp config_simple.example.env .env

# Edit .env with your values (only Databricks auth needed)
# - DATABRICKS_HOST=your-workspace-url
# - DATABRICKS_TOKEN=your-token (OR CLIENT_ID/SECRET)
```

### 3. Run the Simple App
```bash
streamlit run streamlit_simple_agent.py
```

## 📋 Configuration

### Required Environment Variables
- `DATABRICKS_HOST`: Your Databricks workspace URL  
- `DATABRICKS_TOKEN`: Personal access token (OR use CLIENT_ID/SECRET)

### Optional Environment Variables
- `LLM_ENDPOINT_NAME`: Model endpoint (default: databricks-gpt-oss-20b)
- `SYSTEM_PROMPT`: Custom system prompt
- `UC_TOOL_NAMES`: Comma-separated UC function names for tools
- `VECTOR_SEARCH_INDEX_NAMES`: Comma-separated vector search index names

## 🔧 How It Works

### State Management
- **Conversation History**: Stored in Streamlit session state
- **Conversation ID**: Generated per session, not persistent
- **Message History**: Available during the session only

### Benefits
- ✅ **No Database Setup**: Works immediately without database configuration
- ✅ **Lightweight**: Minimal dependencies  
- ✅ **Fast Startup**: No database connections or schema setup
- ✅ **Simple Deployment**: Just needs Databricks access

### Limitations  
- ❌ **No Persistence**: Conversations lost when browser refreshes/closes
- ❌ **No Cross-Session Memory**: Each session is independent
- ❌ **No Thread Management**: Cannot resume previous conversations

## 🎮 Usage

1. **Start App**: Run `streamlit run streamlit_simple_agent.py`
2. **Chat**: Type messages in the chat input
3. **Examples**: Click sidebar examples to try pre-made questions
4. **New Conversation**: Click "New Conversation" to reset
5. **Clear Chat**: Click "Clear Chat" to clear current session

## 🔄 Switching Between Versions

### Use Simple Version When:
- Testing or development
- Don't need conversation persistence  
- Want quick setup without database
- Minimal infrastructure requirements

### Use Main Version When:
- Production deployment
- Need persistent conversations
- Want thread-based memory
- Multiple users sharing conversations

## 📁 Files in Simple Version

- `agent_simple.py` - Simplified agent without database dependencies
- `streamlit_simple_agent.py` - Streamlit app for simple agent  
- `requirements_simple.txt` - Minimal dependencies
- `config_simple.example.env` - Simple configuration template
- `README_SIMPLE.md` - This file

## 🚫 Not Included in Simple Version

- No PostgreSQL/Lakebase dependencies
- No langgraph or checkpoint persistence
- No mlflow integration  
- No complex connection pooling
- No database schema setup
