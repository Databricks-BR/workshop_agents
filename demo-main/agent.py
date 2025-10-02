import json
import logging
import os
import time
import urllib.parse
import uuid
from threading import Lock
from typing import Annotated, Any, Dict, Generator, List, Optional, Sequence, TypedDict

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    DatabricksFunctionClient,
    UCFunctionToolkit,
)
from databricks.sdk import WorkspaceClient
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


############################################
# Define your LLM endpoint and system prompt
############################################
# Configuration from environment variables with fallback defaults
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-3-7-sonnet")

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Use the available tools to answer questions.")

# Try to load .env file if available
try:
    from dotenv import load_dotenv
    if load_dotenv():
        logger.info("✅ Loaded environment variables from .env file")
except ImportError:
    logger.debug("python-dotenv not available, using system environment variables only")

# Lakebase configuration from environment variables (validation happens at agent creation)
def validate_databricks_auth():
    """Validate Databricks authentication configuration to avoid conflicts"""
    token = os.getenv("DATABRICKS_TOKEN")
    client_id = os.getenv("DATABRICKS_CLIENT_ID")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    host = os.getenv("DATABRICKS_HOST")
    
    # Check if we have the required host
    if not host:
        raise ValueError("DATABRICKS_HOST is required")
    
    # Count configured auth methods
    has_token = bool(token and token.strip())
    has_oauth = bool(client_id and client_id.strip() and client_secret and client_secret.strip())
    
    if has_token and has_oauth:
        logger.error("❌ Multiple Databricks authentication methods detected!")
        logger.error("Choose ONE authentication method:")
        logger.error("  Option 1 - Personal Access Token:")
        logger.error("    DATABRICKS_TOKEN=your-token")
        logger.error("    # Remove or comment out CLIENT_ID/CLIENT_SECRET")
        logger.error("  Option 2 - Service Principal OAuth:")
        logger.error("    DATABRICKS_CLIENT_ID=your-client-id")
        logger.error("    DATABRICKS_CLIENT_SECRET=your-client-secret")
        logger.error("    # Remove or comment out DATABRICKS_TOKEN")
        raise ValueError("Multiple authentication methods configured. Use either TOKEN or CLIENT_ID/CLIENT_SECRET, not both.")
    
    if not has_token and not has_oauth:
        logger.error("❌ No Databricks authentication configured!")
        logger.error("Choose ONE authentication method:")
        logger.error("  Option 1 - Personal Access Token (simpler):")
        logger.error("    DATABRICKS_TOKEN=your-personal-access-token")
        logger.error("  Option 2 - Service Principal OAuth (recommended for production):")
        logger.error("    DATABRICKS_CLIENT_ID=your-service-principal-client-id")
        logger.error("    DATABRICKS_CLIENT_SECRET=your-service-principal-secret")
        raise ValueError("No Databricks authentication configured")
    
    if has_token:
        logger.info("✅ Using Personal Access Token authentication")
        # Clear OAuth env vars to avoid conflicts
        if os.getenv("DATABRICKS_CLIENT_ID"):
            del os.environ["DATABRICKS_CLIENT_ID"]
        if os.getenv("DATABRICKS_CLIENT_SECRET"):
            del os.environ["DATABRICKS_CLIENT_SECRET"]
    elif has_oauth:
        logger.info("✅ Using OAuth Service Principal authentication")
        # Clear token to avoid conflicts
        if os.getenv("DATABRICKS_TOKEN"):
            del os.environ["DATABRICKS_TOKEN"]

def get_lakebase_config():
    """Get Lakebase configuration with validation"""
    # First validate Databricks authentication to avoid conflicts
    validate_databricks_auth()
    
    config = {
        "instance_name": os.getenv("LAKEBASE_INSTANCE_NAME"),
        "conn_host": os.getenv("LAKEBASE_HOST"),
        "conn_db_name": os.getenv("LAKEBASE_DB_NAME", "databricks_postgres"),
        "conn_ssl_mode": os.getenv("LAKEBASE_SSL_MODE", "require"),
    }
    
    # Validate required variables
    missing_vars = []
    if not config["instance_name"]:
        missing_vars.append("LAKEBASE_INSTANCE_NAME")
    if not config["conn_host"]:
        missing_vars.append("LAKEBASE_HOST")
    
    if missing_vars:
        logger.error("❌ Missing required Lakebase environment variables:")
        for var in missing_vars:
            logger.error(f"  • {var}")
        logger.error("\n💡 To fix this:")
        logger.error("  1. Copy config.example.env to .env")
        logger.error("  2. Update the values in .env with your configuration")
        logger.error("  3. Or set environment variables directly:")
        for var in missing_vars:
            logger.error(f"     export {var}=your-value-here")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return config

# Configuration will be printed when agent is created

###############################################################################
## Define tools for your agent,enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/en/generative-ai/agent-framework/agent-tool.html
###############################################################################

tools = []

# Example UC tools; add your own as needed
UC_TOOL_NAMES: list[str] = ['ericos_catalog.default.chat_with_sales']
if UC_TOOL_NAMES:
    uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
    tools.extend(uc_toolkit.tools)

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# List to store vector search tool instances for unstructured retrieval.
VECTOR_SEARCH_TOOLS = []

# To add vector search retriever tools,
# use VectorSearchRetrieverTool and create_tool_info,
# then append the result to TOOL_INFOS.
# Example:
# VECTOR_SEARCH_TOOLS.append(
#     VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )

tools.extend(VECTOR_SEARCH_TOOLS)

#####################
## Define agent logic
#####################


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    custom_inputs: Optional[dict[str, Any]]
    custom_outputs: Optional[dict[str, Any]]


class CredentialConnection(psycopg.Connection):
    """Custom connection class that generates fresh OAuth tokens with caching."""
    
    workspace_client = None
    instance_name = None
    
    # Cache attributes
    _cached_credential = None
    _cache_timestamp = None
    _cache_duration = 3000  # 50 minutes in seconds (50 * 60)
    _cache_lock = Lock()
    
    @classmethod
    def connect(cls, conninfo='', **kwargs):
        """Override connect to inject OAuth token with 50-minute caching"""
        if cls.workspace_client is None or cls.instance_name is None:
            raise ValueError("workspace_client and instance_name must be set on CredentialConnection class")
        
        # Get cached or fresh credential and append the new password to kwargs
        credential_token = cls._get_cached_credential()
        kwargs['password'] = credential_token
        
        # Call the superclass's connect method with updated kwargs
        return super().connect(conninfo, **kwargs)
    
    @classmethod
    def _get_cached_credential(cls):
        """Get credential from cache or generate a new one if cache is expired"""
        with cls._cache_lock:
            current_time = time.time()
            
            # Check if we have a valid cached credential
            if (cls._cached_credential is not None and 
                cls._cache_timestamp is not None and 
                current_time - cls._cache_timestamp < cls._cache_duration):
                return cls._cached_credential
            
            # Generate new credential
            credential = cls.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[cls.instance_name]
            )
            
            # Cache the new credential
            cls._cached_credential = credential.token
            cls._cache_timestamp = current_time
            
            return cls._cached_credential


class LangGraphResponsesAgent(ResponsesAgent):
    """Stateful agent using ResponsesAgent with Lakebase PostgreSQL checkpointing.
    
    Features:
    - Connection pooling with credential rotation and caching
    - Thread-based conversation state persistence
    - Tool support with UC functions
    """

    def __init__(self, lakebase_config: dict[str, Any]):
        logger.info("🚀 Initializing LangGraphResponsesAgent...")
        
        # Print configuration
        logger.info("🔧 Agent Configuration:")
        logger.info(f"  LLM_ENDPOINT_NAME: {LLM_ENDPOINT_NAME}")
        logger.info(f"  SYSTEM_PROMPT: {SYSTEM_PROMPT[:50]}{'...' if len(SYSTEM_PROMPT) > 50 else ''}")
        logger.info(f"  LAKEBASE_INSTANCE_NAME: {lakebase_config['instance_name']}")
        logger.info(f"  LAKEBASE_HOST: {lakebase_config['conn_host']}")
        logger.info(f"  LAKEBASE_DB_NAME: {lakebase_config['conn_db_name']}")
        logger.info(f"  LAKEBASE_SSL_MODE: {lakebase_config['conn_ssl_mode']}")
        
        # Print environment variables being used
        logger.info("📋 Environment Variables:")
        logger.info(f"  DATABRICKS_HOST: {os.getenv('DATABRICKS_HOST', 'Not set')}")
        logger.info(f"  DATABRICKS_CLIENT_ID: {'***' + os.getenv('DATABRICKS_CLIENT_ID', 'Not set')[-4:] if os.getenv('DATABRICKS_CLIENT_ID') else 'Not set'}")
        logger.info(f"  DATABRICKS_CLIENT_SECRET: {'***' if os.getenv('DATABRICKS_CLIENT_SECRET') else 'Not set'}")
        logger.info(f"  DATABRICKS_TOKEN: {'***' if os.getenv('DATABRICKS_TOKEN') else 'Not set'}")
        logger.info(f"  DB_POOL_MIN_SIZE: {os.getenv('DB_POOL_MIN_SIZE', '1')}")
        logger.info(f"  DB_POOL_MAX_SIZE: {os.getenv('DB_POOL_MAX_SIZE', '10')}")
        logger.info(f"  DB_TOKEN_CACHE_MINUTES: {os.getenv('DB_TOKEN_CACHE_MINUTES', '50')}")
        
        self.lakebase_config = lakebase_config
        
        logger.info("🔗 Creating Databricks workspace client...")
        self.workspace_client = WorkspaceClient()
        logger.info("✅ Workspace client created successfully")
        
        # Model and tools
        logger.info(f"🤖 Initializing ChatDatabricks with endpoint: {LLM_ENDPOINT_NAME}")
        self.model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
        self.system_prompt = SYSTEM_PROMPT
        self.model_with_tools = self.model.bind_tools(tools) if tools else self.model
        logger.info(f"✅ Model initialized with {len(tools)} tools")
        
        # Connection pool configuration
        self.pool_min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
        self.pool_max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
        self.pool_timeout = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))
        
        # Token cache duration (in minutes, can be overridden via env var)
        cache_duration_minutes = int(os.getenv("DB_TOKEN_CACHE_MINUTES", "50"))
        CredentialConnection._cache_duration = cache_duration_minutes * 60
        
        logger.info("🔄 Creating database connection pool...")
        # Initialize the connection pool with rotating credentials
        self._connection_pool = self._create_rotating_pool()
        logger.info("✅ Connection pool initialized successfully")
        
        # Set up checkpointer database schema once during initialization
        self._setup_checkpointer_schema()
        
        mlflow.langchain.autolog()
        logger.info("🎯 LangGraphResponsesAgent initialization complete!")

    def _get_username(self) -> str:
        """Get the username for database connection from environment variable"""
        client_id = os.getenv("DATABRICKS_CLIENT_ID")
        if client_id:
            logger.info(f"✅ Using DATABRICKS_CLIENT_ID as username: ***{client_id[-4:]}")
            return client_id
        else:
            logger.warning("⚠️  DATABRICKS_CLIENT_ID not found, falling back to workspace client")
            try:
                sp = self.workspace_client.current_service_principal.me()
                return sp.application_id
            except Exception:
                user = self.workspace_client.current_user.me()
                return user.user_name

    def _create_rotating_pool(self) -> ConnectionPool:
        """Create a connection pool that automatically rotates credentials with caching"""
        # Set the workspace client and instance name on the custom connection class
        logger.info("🔑 Setting up credential rotation...")
        CredentialConnection.workspace_client = self.workspace_client
        CredentialConnection.instance_name = self.lakebase_config["instance_name"]
        
        logger.info("👤 Getting database username...")
        username = self._get_username()
        logger.info(f"✅ Using username: {username}")
        
        host = self.lakebase_config["conn_host"]
        database = self.lakebase_config.get("conn_db_name", "databricks_postgres")
        
        connection_info = f"dbname={database} user={username} host={host} sslmode=require"
        logger.info(f"🔗 Connection string: dbname={database} user={username} host={host[:20]}... sslmode=require")
        
        logger.info(f"🏊 Creating connection pool (min={self.pool_min_size}, max={self.pool_max_size}, timeout={self.pool_timeout}s)...")
        
        # Create pool with custom connection class
        pool = ConnectionPool(
            conninfo=connection_info,
            connection_class=CredentialConnection,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            timeout=self.pool_timeout,
            open=True,
            kwargs={
                "autocommit": True, # Required for the .setup() method to properly commit the checkpoint tables to the database
                "row_factory": dict_row, # Required because the PostgresSaver implementation accesses database rows using dictionary-style syntax
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
        )
        
        # Test the pool
        logger.info("🧪 Testing connection pool...")
        try:
            with pool.connection() as conn:
                logger.info("✅ Got connection from pool")
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    logger.info(f"✅ Database test query successful: {result}")
            logger.info(
                f"🎉 Connection pool with rotating credentials created successfully "
                f"(min={self.pool_min_size}, max={self.pool_max_size}, "
                f"token_cache={CredentialConnection._cache_duration / 60:.0f} minutes)"
            )
        except Exception as e:
            logger.error(f"❌ Connection pool test failed: {e}")
            logger.error(f"❌ Connection details - Host: {host}, Database: {database}, User: {username}")
            pool.close()
            raise ConnectionError(f"Failed to create connection pool: {e}")
        
        return pool
    
    def _setup_checkpointer_schema(self):
        """Set up the checkpointer database schema once during initialization"""
        logger.info("🗃️  Setting up checkpointer database schema...")
        try:
            with self.get_connection() as conn:
                checkpointer = PostgresSaver(conn)
                checkpointer.setup()
                logger.info("✅ Checkpointer database schema created successfully")
        except Exception as e:
            logger.warning(f"⚠️  Checkpointer schema setup issue (might already exist): {e}")
            # This is usually fine - the schema might already exist
    
    @contextmanager
    def get_connection(self):
        """Context manager to get a connection from the pool"""
        with self._connection_pool.connection() as conn:
            yield conn
    
    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert from LangChain messages to Responses API format"""
        responses = []
        for message in messages:
            message_dict = message.model_dump()
            msg_type = message_dict["type"]
            
            if msg_type == "ai":
                if tool_calls := message_dict.get("tool_calls"):
                    for tool_call in tool_calls:
                        responses.append(
                            self.create_function_call_item(
                                id=message_dict.get("id") or str(uuid.uuid4()),
                                call_id=tool_call["id"],
                                name=tool_call["name"],
                                arguments=json.dumps(tool_call["args"]),
                            )
                        )
                else:
                    responses.append(
                        self.create_text_output_item(
                            text=message_dict.get("content", ""),
                            id=message_dict.get("id") or str(uuid.uuid4()),
                        )
                    )
            elif msg_type == "tool":
                responses.append(
                    self.create_function_call_output_item(
                        call_id=message_dict["tool_call_id"],
                        output=message_dict["content"],
                    )
                )
            elif msg_type == "human":
                responses.append({
                    "role": "user",
                    "content": message_dict.get("content", "")
                })
        
        return responses
    
    def _create_graph(self, checkpointer: PostgresSaver):
        """Create the LangGraph workflow"""
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "continue"
            return "end"
        
        if self.system_prompt:
            preprocessor = RunnableLambda(
                lambda state: [{"role": "system", "content": self.system_prompt}] + state["messages"]
            )
        else:
            preprocessor = RunnableLambda(lambda state: state["messages"])
        
        model_runnable = preprocessor | self.model_with_tools
        
        def call_model(state: AgentState, config: RunnableConfig):
            response = model_runnable.invoke(state, config)
            return {"messages": [response]}
        
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", RunnableLambda(call_model))
        
        if tools:
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {"continue": "tools", "end": END}
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)
        
        workflow.set_entry_point("agent")
        
        return workflow.compile(checkpointer=checkpointer)
    
    def get_checkpoint_history(self, thread_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve checkpoint history for a thread.
        
        Args:
            thread_id: The thread identifier
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint information including checkpoint_id, timestamp, and next nodes
        """
        config = {"configurable": {"thread_id": thread_id}}
        with self.get_connection() as conn:
            checkpointer = PostgresSaver(conn)
            graph = self._create_graph(checkpointer)
            history = []
            
            for state in graph.get_state_history(config):
                if len(history) >= limit:
                    break
                history.append({
                    "checkpoint_id": state.config["configurable"]["checkpoint_id"],
                    "thread_id": thread_id,
                    "timestamp": state.created_at,
                    "next_nodes": state.next,
                    "message_count": len(state.values.get("messages", [])),
                    # Include last message summary for context
                    "last_message": self._get_last_message_summary(state.values.get("messages", []))
                })
            return history
    
    def _get_last_message_summary(self, messages: List[Any]) -> Optional[str]:
        """Get a snippet of the last message for checkpoint identification"""
        if not messages:
            return None
        last_msg = messages[-1]
        content = getattr(last_msg, "content", "")
        return content[:100] if content else None
    
    def update_checkpoint_state(self, thread_id: str, checkpoint_id: str,
                            new_messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Update state at a specific checkpoint (used for modifying conversation history).
        
        Args:
            thread_id: The thread identifier
            checkpoint_id: The checkpoint to update
            new_messages: Optional new messages to set at this checkpoint
            
        Returns:
            New checkpoint configuration including the new checkpoint_id
        """
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id
            }
        }
        
        with self.get_connection() as conn:
            checkpointer = PostgresSaver(conn)
            graph = self._create_graph(checkpointer)
            
            # Prepare the values to update
            values = {}
            if new_messages:
                cc_msgs = self.prep_msgs_for_cc_llm(new_messages)
                values["messages"] = cc_msgs
            
            # Update the state (creates a new checkpoint)
            new_config = graph.update_state(config, values=values)
            
            return {
                "thread_id": thread_id,
                "checkpoint_id": new_config["configurable"]["checkpoint_id"],
                "parent_checkpoint_id": checkpoint_id
            }
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Non-streaming prediction"""
        # The same thread_id is used by BOTH predict() and predict_stream()
        ci = dict(request.custom_inputs or {})
        if "thread_id" not in ci:
            ci["thread_id"] = str(uuid.uuid4())
        request.custom_inputs = ci

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        # Include thread_id and checkpoint_id in custom outputs
        custom_outputs = {
            "thread_id": ci["thread_id"]
        }
        if "checkpoint_id" in ci:
            custom_outputs["parent_checkpoint_id"] = ci["checkpoint_id"]
            
        try:
            history = self.get_checkpoint_history(ci["thread_id"], limit=1)
            if history:
                custom_outputs["checkpoint_id"] = history[0]["checkpoint_id"]
        except Exception as e:
            logger.warning(f"Could not retrieve new checkpoint_id: {e}")
            
        return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)
    
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction with PostgreSQL checkpoint branching support.
        
        Accepts in custom_inputs:
        - thread_id: Conversation thread identifier for session
        - checkpoint_id (optional): Checkpoint to resume from (for branching)
        """
        # Get thread ID and checkpoint ID from custom inputs
        custom_inputs = request.custom_inputs or {}
        thread_id = custom_inputs.get("thread_id", str(uuid.uuid4()))  # generate new thread ID if one is not passed in
        checkpoint_id = custom_inputs.get("checkpoint_id")  # Optional for branching
        
        # Convert incoming Responses messages to LangChain format
        langchain_msgs = self.prep_msgs_for_cc_llm([i.model_dump() for i in request.input])
        
        # Build checkpoint configuration
        checkpoint_config = {"configurable": {"thread_id": thread_id}}
        
        # If checkpoint_id is provided, we're branching from that checkpoint
        if checkpoint_id:
            checkpoint_config["configurable"]["checkpoint_id"] = checkpoint_id
            logger.info(f"Branching from checkpoint: {checkpoint_id} in thread: {thread_id}")
        
        # DATABASE CONNECTION POOLING LOGIC FOLLOWS
        
        # Use connection from pool
        with self.get_connection() as conn:            
            # Create checkpointer and graph
            checkpointer = PostgresSaver(conn)
            graph = self._create_graph(checkpointer)
            
            # Stream the graph execution
            for event in graph.stream(
                {"messages": langchain_msgs},
                checkpoint_config,
                stream_mode=["updates", "messages"]
            ):
                if event[0] == "updates":
                    for node_data in event[1].values():
                        for item in self._langchain_to_responses(node_data["messages"]):
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item=item
                            )
                # Stream message chunks for real-time text generation
                elif event[0] == "messages":
                    try:
                        chunk = event[1][0]
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            yield ResponsesAgentStreamEvent(
                                **self.create_text_delta(
                                    delta=chunk.content,
                                    item_id=chunk.id
                                ),
                            )
                    except Exception as e:
                        logger.error(f"Error streaming chunk: {e}")


# ----- Export model -----
def create_agent():
    """Create the agent instance with proper error handling"""
    try:
        # Get and validate configuration
        lakebase_config = get_lakebase_config()
        return LangGraphResponsesAgent(lakebase_config)
    except ValueError as e:
        # This is our validation error with user-friendly message
        raise e
    except Exception as e:
        logger.error(f"Failed to initialize LangGraphResponsesAgent: {e}")
        logger.error("Please ensure you have:")
        logger.error("1. Databricks workspace credentials configured")
        logger.error("2. Access to the specified Lakebase instance")
        logger.error("3. All required packages installed: pip install -r requirements.txt")
        raise

# Create agent instance - will be initialized when imported
AGENT = None

def get_agent():
    """Lazy initialization of agent"""
    global AGENT
    if AGENT is None:
        AGENT = create_agent()
        mlflow.models.set_model(AGENT)
    return AGENT
