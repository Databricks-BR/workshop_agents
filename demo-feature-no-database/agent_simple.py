import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Any

# Import only essential libraries (no database dependencies)
from databricks_langchain import ChatDatabricks, DatabricksFunctionClient, UCFunctionToolkit
from databricks.sdk import WorkspaceClient  
from langchain_core.messages import HumanMessage, AIMessage

# Try to load .env file if available
try:
    from dotenv import load_dotenv
    if load_dotenv():
        logging.info("✅ Loaded environment variables from .env file")
except ImportError:
    logging.debug("python-dotenv not available, using system environment variables only")

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Configuration from environment variables
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-gpt-oss-20b")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant. Use the available tools to answer questions.")

############################################
# Define tools
############################################

tools = []

# Example UC tools; add your own as needed
UC_TOOL_NAMES: list[str] = []
UC_TOOL_NAMES_ENV = os.getenv("UC_TOOL_NAMES", "")
if UC_TOOL_NAMES_ENV:
    UC_TOOL_NAMES = [name.strip() for name in UC_TOOL_NAMES_ENV.split(",") if name.strip()]

if UC_TOOL_NAMES:
    logger.info(f"🔧 Loading UC Tools: {UC_TOOL_NAMES}")
    try:
        uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
        tools.extend(uc_toolkit.tools)
        logger.info(f"✅ Loaded {len(uc_toolkit.tools)} UC tools")
    except Exception as e:
        logger.error(f"❌ Error loading UC tools: {e}")

# Use Databricks vector search indexes as tools
# See https://docs.databricks.com/en/generative-ai/agent-framework/unstructured-retrieval-tools.html#locally-develop-vector-search-retriever-tools-with-ai-bridge
# List to store vector search tool instances for unstructured retrieval.
VECTOR_SEARCH_TOOLS = []

# To add vector search retriever tools, set VECTOR_SEARCH_INDEX_NAMES environment variable
# Example: VECTOR_SEARCH_INDEX_NAMES="index1,index2,index3"
VECTOR_SEARCH_INDEX_NAMES_ENV = os.getenv("VECTOR_SEARCH_INDEX_NAMES", "")
if VECTOR_SEARCH_INDEX_NAMES_ENV:
    try:
        from databricks_langchain import VectorSearchRetrieverTool
        
        index_names = [name.strip() for name in VECTOR_SEARCH_INDEX_NAMES_ENV.split(",") if name.strip()]
        logger.info(f"🔍 Loading Vector Search Tools: {index_names}")
        
        for index_name in index_names:
            try:
                vector_tool = VectorSearchRetrieverTool(
                    index_name=index_name,
                    # filters="..."  # Add filters if needed
                )
                VECTOR_SEARCH_TOOLS.append(vector_tool)
                logger.info(f"✅ Loaded vector search tool for index: {index_name}")
            except Exception as e:
                logger.error(f"❌ Error loading vector search tool for {index_name}: {e}")
                
    except ImportError:
        logger.warning("⚠️ VectorSearchRetrieverTool not available, skipping vector search tools")
    except Exception as e:
        logger.error(f"❌ Error setting up vector search tools: {e}")

tools.extend(VECTOR_SEARCH_TOOLS)

logger.info(f"🛠️ Total tools loaded: {len(tools)}")
if tools:
    tool_names = [getattr(tool, 'name', str(tool)) for tool in tools]
    logger.info(f"📋 Available tools: {tool_names}")

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

class SimpleStatefulAgent:
    """
    Simplified stateful agent that uses in-memory conversation storage
    instead of database persistence. Conversation state is managed
    by the calling application (e.g., Streamlit session state).
    """

    def __init__(self):
        logger.info("🚀 Initializing SimpleStatefulAgent (no database)...")
        
        # Validate authentication first
        validate_databricks_auth()
        
        # Print configuration
        logger.info("🔧 Agent Configuration:")
        logger.info(f"  LLM_ENDPOINT_NAME: {LLM_ENDPOINT_NAME}")
        logger.info(f"  SYSTEM_PROMPT: {SYSTEM_PROMPT[:50]}{'...' if len(SYSTEM_PROMPT) > 50 else ''}")
        
        # Print environment variables being used
        logger.info("📋 Environment Variables:")
        logger.info(f"  DATABRICKS_HOST: {os.getenv('DATABRICKS_HOST', 'Not set')}")
        logger.info(f"  DATABRICKS_CLIENT_ID: {'***' + os.getenv('DATABRICKS_CLIENT_ID', 'Not set')[-4:] if os.getenv('DATABRICKS_CLIENT_ID') else 'Not set'}")
        logger.info(f"  DATABRICKS_CLIENT_SECRET: {'***' if os.getenv('DATABRICKS_CLIENT_SECRET') else 'Not set'}")
        logger.info(f"  DATABRICKS_TOKEN: {'***' if os.getenv('DATABRICKS_TOKEN') else 'Not set'}")
        
        # Initialize model
        logger.info(f"🤖 Initializing ChatDatabricks with endpoint: {LLM_ENDPOINT_NAME}")
        self.model = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
        self.system_prompt = SYSTEM_PROMPT
        
        # Initialize tools if available
        self.tools = tools
        if self.tools:
            logger.info(f"🛠️ Agent initialized with {len(self.tools)} tools")
            tool_names = [getattr(tool, 'name', str(tool)) for tool in self.tools]
            logger.info(f"📋 Tools: {tool_names}")
            
            # Update system prompt to include tool information
            tool_list = "\n".join([f"- {name}: {getattr(tool, 'description', 'No description available')}" for name, tool in zip(tool_names, self.tools)])
            self.system_prompt = f"{SYSTEM_PROMPT}\n\nYou have access to the following tools:\n{tool_list}\n\nWhen a user asks about available tools or mentions a tool name, you should acknowledge if you have access to it."
        else:
            logger.info("🤖 Agent initialized without tools (simple mode)")
            self.system_prompt = SYSTEM_PROMPT

    def predict(self, messages: List[Dict[str, str]], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response based on the conversation messages.
        
        Args:
            messages: List of conversation messages in format [{"role": "user|assistant", "content": "..."}]
            conversation_id: Optional conversation identifier (for compatibility, not used in simple version)
        
        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"🎯 Processing request with {len(messages)} messages")
            
            # Convert messages to LangChain format
            langchain_messages = []
            
            # Add system prompt if provided
            if self.system_prompt:
                langchain_messages.append(HumanMessage(content=f"System: {self.system_prompt}"))
            
            # Convert conversation messages
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            # Generate response
            logger.info("🔄 Calling LLM model...")
            # If tools are available, bind them to the model
            if self.tools:
                logger.debug(f"Using model with {len(self.tools)} tools")
                model_with_tools = self.model.bind_tools(self.tools)
                response = model_with_tools.invoke(langchain_messages)
                
                # Check if the model wants to use tools
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"🛠️ Model wants to use {len(response.tool_calls)} tool(s)")
                    
                    # Execute the tool calls
                    from langchain_core.messages import ToolMessage
                    messages_with_tools = langchain_messages + [response]
                    
                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        logger.info(f"🔧 Executing tool: {tool_call['name']} with args: {tool_call['args']}")
                        
                        # Find the tool by name
                        tool_to_use = None
                        for tool in self.tools:
                            if tool.name == tool_call['name']:
                                tool_to_use = tool
                                break
                        
                        if tool_to_use:
                            try:
                                # Execute the tool
                                tool_result = tool_to_use.invoke(tool_call['args'])
                                logger.info(f"✅ Tool {tool_call['name']} result: {str(tool_result)[:100]}...")
                                
                                # Add tool result to messages
                                tool_message = ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_call['id']
                                )
                                messages_with_tools.append(tool_message)
                                
                            except Exception as e:
                                logger.error(f"❌ Error executing tool {tool_call['name']}: {e}")
                                tool_message = ToolMessage(
                                    content=f"Error executing tool: {str(e)}",
                                    tool_call_id=tool_call['id']
                                )
                                messages_with_tools.append(tool_message)
                        else:
                            logger.error(f"❌ Tool {tool_call['name']} not found")
                    
                    # Get final response from model after tool execution
                    logger.info("🔄 Getting final response after tool execution...")
                    final_response = model_with_tools.invoke(messages_with_tools)
                    response = final_response
                    
            else:
                logger.debug("Using model without tools")
                response = self.model.invoke(langchain_messages)
            
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Process and format the response properly
            raw_response = response.content
            logger.debug(f"Raw GPT response: {repr(raw_response[:200])}...")  # Log first 200 chars for debugging
            logger.debug(f"Response type: {type(response)}, has tool_calls: {hasattr(response, 'tool_calls')}")
            
            # Handle tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"🛠️ Model made {len(response.tool_calls)} tool calls")
                # The model wants to use tools - we should process the tool calls
                # For now, let's see if there's any text content alongside the tool calls
                formatted_response = self._format_response(raw_response) 
                if not formatted_response or formatted_response.strip() == "":
                    formatted_response = f"I'm using {len(response.tool_calls)} tool(s) to help answer your question."
            else:
                # Handle different response formats from GPT model
                formatted_response = self._format_response(raw_response)
            
            # Only provide fallback if we truly have no content AND no tool calls
            if not formatted_response or formatted_response.strip() == "":
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    formatted_response = "Processing your request with tools..."
                elif self.tools:
                    formatted_response = "I'm processing your request. The tools are available but not being used for this query."
                else:
                    formatted_response = "I'm processing your request."
            
            logger.debug(f"Formatted response: {repr(formatted_response[:200])}...")  # Log formatted version
            
            result = {
                "response": formatted_response,
                "conversation_id": conversation_id,
                "message_count": len(messages) + 1,  # +1 for the new response
                "model_endpoint": LLM_ENDPOINT_NAME,
                "tools_used": len(self.tools) if self.tools else 0
            }
            
            logger.info(f"✅ Generated response ({len(formatted_response)} characters)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise

    def _format_response(self, raw_response) -> str:
        """
        Format the response from GPT model to handle proper line breaks and JSON formatting.
        
        Args:
            raw_response: Raw response content from the model (can be string, list, or dict)
            
        Returns:
            Properly formatted response text
        """
        try:
            logger.debug(f"Formatting response of type: {type(raw_response)}")
            
            # Handle different input types first
            if isinstance(raw_response, list):
                # Check if it's a list of structured response objects from GPT
                text_contents = []
                for item in raw_response:
                    if isinstance(item, dict) and 'type' in item:
                        # Handle GPT's structured response format
                        if item['type'] == 'text' and 'text' in item:
                            text_contents.append(item['text'])
                        elif item['type'] == 'reasoning':
                            # Skip reasoning objects completely
                            logger.debug("Skipping reasoning content")
                            continue
                        # Skip other types we don't want to display
                    elif isinstance(item, str):
                        text_contents.append(item)
                    else:
                        # Convert other items to string
                        text_contents.append(str(item))
                
                if text_contents:
                    raw_response = '\n'.join(text_contents)
                else:
                    # Fallback: join all items as strings, but filter out tool-related content
                    filtered_items = []
                    for item in raw_response:
                        item_str = str(item)
                        # Skip obvious tool-related content
                        if not any(keyword in item_str.lower() for keyword in ['tool_call', 'function_call', 'tool_use']):
                            filtered_items.append(item_str)
                    raw_response = '\n'.join(filtered_items) if filtered_items else str(raw_response)
                    
            elif isinstance(raw_response, dict):
                # If it's already a dict, try to extract content
                if 'type' in raw_response:
                    if raw_response['type'] == 'text' and 'text' in raw_response:
                        # Handle single GPT text object
                        return self._process_text_formatting(raw_response['text'])
                    elif raw_response['type'] == 'reasoning':
                        # This is internal reasoning - filter it out, but don't return empty
                        # Let the process continue to see if there are other content types
                        logger.debug("Filtering out reasoning content from response")
                        return ""  # Return empty to trigger higher-level fallback
                else:
                    # Try standard content fields
                    content_fields = ['message', 'content', 'text', 'response', 'answer']
                    for field in content_fields:
                        if field in raw_response:
                            return self._process_text_formatting(raw_response[field])
                    # If no specific field found, check if it contains reasoning
                    raw_str = str(raw_response)
                    if any(keyword in raw_str.lower() for keyword in ['reasoning', 'tool_call', 'function_call', 'tool_use']):
                        logger.debug("Filtering out tool/reasoning content from response")
                        # Return empty to let the higher-level logic handle it
                        return ""
                    else:
                        raw_response = raw_str
            elif not isinstance(raw_response, str):
                raw_response = str(raw_response)
            
            # Now raw_response should be a string
            # If the response looks like JSON, try to parse it
            if raw_response.strip().startswith('{') and raw_response.strip().endswith('}'):
                try:
                    parsed_json = json.loads(raw_response)
                    # If it's a JSON response, extract the actual message content
                    if isinstance(parsed_json, dict):
                        # Look for common response fields
                        content_fields = ['message', 'content', 'text', 'response', 'answer']
                        for field in content_fields:
                            if field in parsed_json:
                                return self._process_text_formatting(parsed_json[field])
                        
                        # If no specific field found, convert the whole JSON to readable text  
                        return self._process_text_formatting(str(parsed_json))
                except json.JSONDecodeError:
                    # Not valid JSON, treat as regular text
                    pass
            
            # Process as regular text
            return self._process_text_formatting(raw_response)
            
        except Exception as e:
            logger.warning(f"Error formatting response: {e}, returning raw response as string")
            return self._process_text_formatting(str(raw_response))
    
    def _process_text_formatting(self, text) -> str:
        """
        Process text to handle proper line breaks and formatting.
        
        Args:
            text: Text to process (can be string, list, or other types)
            
        Returns:
            Properly formatted text with correct line breaks
        """
        if not text:
            return ""
        
        # Handle different input types
        if isinstance(text, list):
            # If it's a list, join the elements
            if all(isinstance(item, str) for item in text):
                text = '\n'.join(text)
            else:
                # Convert non-string items to string first
                text = '\n'.join(str(item) for item in text)
        elif not isinstance(text, str):
            # Convert other types to string
            text = str(text)
        
        # Handle escaped newlines and characters
        text = text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        
        # Clean up extra whitespace but preserve intentional line breaks
        lines = text.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        # Remove excessive blank lines (more than 2 consecutive)
        result_lines = []
        blank_count = 0
        
        for line in cleaned_lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:  # Allow up to 2 blank lines
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines).strip()

    def predict_with_history(self, new_message: str, conversation_history: List[Dict[str, str]], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Convenience method to add a new message to conversation history and generate response.
        
        Args:
            new_message: The new user message
            conversation_history: Previous conversation messages
            conversation_id: Optional conversation identifier
            
        Returns:
            Dictionary with response and metadata
        """
        # Add the new user message to history
        updated_history = conversation_history + [{"role": "user", "content": new_message}]
        
        # Generate response
        result = self.predict(updated_history, conversation_id)
        
        return result

# Global agent instance
_AGENT_INSTANCE = None

def get_simple_agent():
    """Get the global simple agent instance (lazy initialization)"""
    global _AGENT_INSTANCE
    if _AGENT_INSTANCE is None:
        _AGENT_INSTANCE = SimpleStatefulAgent()
    return _AGENT_INSTANCE

def create_simple_agent():
    """Create a new simple agent instance"""
    return SimpleStatefulAgent()

# For compatibility with the main agent interface
AGENT = get_simple_agent
