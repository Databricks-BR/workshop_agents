import os
import streamlit as st
import json
import uuid
from typing import Dict, Optional
import logging

# Import the agent from the existing agent.py file
from agent import get_agent

# Configuration - you may need to adjust these based on your setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_llm_response(response_data: Dict) -> str:
    """Process the LLM agent response and format it for display"""
    response = ""
    
    try:
        if 'output' in response_data and response_data['output']:
            for output_item in response_data['output']:
                if output_item.get('type') == 'message':
                    content = output_item.get('content', [])
                    for content_item in content:
                        if content_item.get('type') == 'output_text':
                            response += content_item.get('text', '')
                        elif content_item.get('type') == 'function_call':
                            # Handle function calls if they exist
                            function_name = content_item.get('name', 'Unknown Function')
                            response += f"\n**Function Call:** {function_name}\n"
                        elif content_item.get('type') == 'function_output':
                            # Handle function outputs
                            output = content_item.get('output', '')
                            response += f"\n**Function Output:**\n```\n{output}\n```\n"
        
        # If no content found, try to get it from a different structure
        if not response and 'content' in response_data:
            response = str(response_data['content'])
        
        # Fallback to the raw response if nothing else worked
        if not response:
            response = str(response_data)
            
    except Exception as e:
        logger.error(f"Error processing LLM response: {str(e)}")
        response = "Error processing response. Please try again."
    
    return response if response else "No response generated."

def submit_question():
    """Handle form submission"""
    # The processing state will be set when we actually start processing
    pass

def get_checkpoint_conversations():
    """Get all conversation threads with their checkpoint history using LangGraph"""
    if not st.session_state.checkpoint_threads:
        return []
    
    try:
        agent = get_agent()
        conversations = []
        
        for thread_id in st.session_state.checkpoint_threads:
            try:
                history = agent.get_checkpoint_history(thread_id, limit=50)
                if history:
                    # Get title from the first checkpoint with messages
                    title = None
                    for checkpoint in reversed(history):  # Start from oldest
                        if checkpoint.get("last_message"):
                            msg = checkpoint["last_message"]
                            title = msg[:50] + "..." if len(msg) > 50 else msg
                            break
                    
                    conversations.append({
                        "thread_id": thread_id,
                        "title": title or f"Thread {thread_id[:8]}",
                        "checkpoint_count": len(history),
                        "latest_timestamp": history[0]["timestamp"] if history else None,
                        "message_count": history[0]["message_count"] if history else 0
                    })
            except Exception as e:
                logger.warning(f"Error getting history for thread {thread_id}: {e}")
                continue
                
        # Sort by latest timestamp
        conversations.sort(key=lambda x: x["latest_timestamp"] or "", reverse=True)
        return conversations
        
    except Exception as e:
        logger.error(f"Error getting checkpoint conversations: {e}")
        return []

def load_conversation_from_thread(thread_id: str):
    """Load conversation history from LangGraph checkpoints"""
    try:
        agent = get_agent()
        history = agent.get_checkpoint_history(thread_id, limit=50)
        
        if not history:
            st.error("No checkpoint history found for this thread")
            return
            
        # Get the latest checkpoint state to reconstruct conversation
        st.session_state.conversation_thread_id = thread_id
        st.session_state.chat_history = []  # Will be rebuilt from agent responses
        st.session_state.processing = False
        
        # Add thread to our tracking set
        st.session_state.checkpoint_threads.add(thread_id)
        
        st.success(f"Loaded thread {thread_id[:8]} with {len(history)} checkpoints")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error loading conversation: {e}")

def start_new_conversation():
    """Start a completely new conversation"""
    # Just reset the UI state - LangGraph handles the rest
    st.session_state.conversation_thread_id = None
    st.session_state.chat_history = []
    st.session_state.processing = False

def ask_llm_agent(question: str, thread_id: Optional[str] = None) -> tuple[str, str]:
    """Ask the LLM agent a question and return the response with thread_id"""
    try:
        # Prepare the request
        request_data = {
            "input": [{"role": "user", "content": question}]
        }
        
        # Add thread_id if provided for conversation continuity
        if thread_id:
            request_data["custom_inputs"] = {"thread_id": thread_id}
        
        # Call the agent
        agent = get_agent()
        response = agent.predict(request_data)
        
        # Extract thread_id from response
        new_thread_id = response.custom_outputs.get("thread_id") if response.custom_outputs else None
        
        # Process the response
        response_dict = response.model_dump(exclude_none=True)
        formatted_response = process_llm_response(response_dict)
        
        return formatted_response, new_thread_id
        
    except Exception as e:
        logger.error(f"Error in ask_llm_agent: {str(e)}")
        return f"Error: {str(e)}", thread_id

def main():
    # Initialize session state
    if 'conversation_thread_id' not in st.session_state:
        st.session_state.conversation_thread_id = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Initialize checkpoint storage
    if 'checkpoint_threads' not in st.session_state:
        st.session_state.checkpoint_threads = set()  # Track threads we've used

    # Streamlit UI
    st.title("Chatbot powered by LLM Agent")
    st.caption("This chatbot uses a stateful LLM agent with thread-based memory persistence")

    # Display current thread info
    if st.session_state.conversation_thread_id:
        st.sidebar.info(f"💬 Thread {st.session_state.conversation_thread_id[:8]}")
        
        # Show checkpoint count if available
        try:
            agent = get_agent()
            history = agent.get_checkpoint_history(st.session_state.conversation_thread_id, limit=1)
            if history:
                st.sidebar.caption(f"💾 {history[0]['message_count']} messages saved")
        except:
            pass
    else:
        st.sidebar.info("🆕 New conversation")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar with examples
    st.sidebar.title("Examples")
    examples = [
        "What is a stateful agent?",
        "Explain the benefits of using LangGraph",
        "How does thread-based memory work?",
        "What are the advantages of using Lakebase for agent state?",
    ]

    # Add example buttons
    for example in examples:
        if st.sidebar.button(example):
            st.session_state.example_clicked = example

    # User input
    user_input = st.chat_input(
        "Type your message here...", 
        on_submit=submit_question, 
        disabled=st.session_state.processing
    )

    # Handle example clicks
    if hasattr(st.session_state, 'example_clicked'):
        user_input = st.session_state.example_clicked
        delattr(st.session_state, 'example_clicked')

    if user_input and not st.session_state.processing:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get bot response
        with st.chat_message("assistant"):
            st.session_state.processing = True
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Thinking...")

            try:
                logger.info(f"Current thread_id: {st.session_state.conversation_thread_id}")
                
                # Ask the LLM agent
                response, new_thread_id = ask_llm_agent(
                    user_input, 
                    st.session_state.conversation_thread_id
                )
                
                # Update thread_id
                st.session_state.conversation_thread_id = new_thread_id
                logger.info(f"New thread_id: {st.session_state.conversation_thread_id}")
                
                # Display response
                message_placeholder.markdown(response)
                
                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Track this thread in our checkpoint threads
                if st.session_state.conversation_thread_id:
                    st.session_state.checkpoint_threads.add(st.session_state.conversation_thread_id)
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                error_message = f"An error occurred while processing your request: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            
            finally:
                st.session_state.processing = False
                # Force Streamlit to refresh and re-enable the chat input
                st.rerun()

    # Sidebar controls
    st.sidebar.markdown("---")
    
    # LangGraph Checkpoint Management
    st.sidebar.markdown("### 💬 LangGraph Conversations")
    
    if st.sidebar.button("🆕 New Conversation", help="Start new conversation"):
        start_new_conversation()
        st.rerun()
    
    # Show checkpoint-based conversations
    conversations = get_checkpoint_conversations()
    if conversations:
        st.sidebar.markdown("#### 📚 Checkpoint Threads")
        
        for conv in conversations:
            is_current = conv["thread_id"] == st.session_state.conversation_thread_id
            
            # Show conversation with status indicator
            status_icon = "🟢" if is_current else "⚪"
            title = conv["title"]
            msg_count = conv["message_count"]
            checkpoint_count = conv["checkpoint_count"]
            
            if st.sidebar.button(
                f"{status_icon} {title}",
                help=f"Messages: {msg_count} | Checkpoints: {checkpoint_count}",
                disabled=is_current,
                key=f"load_thread_{conv['thread_id']}"
            ):
                load_conversation_from_thread(conv["thread_id"])
        
        # Clear thread tracking
        if st.sidebar.button("🗑️ Clear Thread History"):
            st.session_state.checkpoint_threads = set()
            st.sidebar.success("Thread history cleared!")
            st.rerun()
    else:
        st.sidebar.info("No checkpoint conversations found")
    
    # Display agent info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Agent Info")
    st.sidebar.text("• LangGraph checkpoint persistence")
    st.sidebar.text("• Thread-based conversations")
    st.sidebar.text("• Lakebase state storage")
    st.sidebar.text("• Time-travel capabilities")
    st.sidebar.text("• Claude 3.5 Sonnet model")

if __name__ == "__main__":
    main()
