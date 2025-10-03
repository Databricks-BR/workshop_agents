import os
import streamlit as st
import json
import uuid
from typing import Dict, Optional
import logging

# Import the simple agent
from agent_simple import get_simple_agent

# Configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_simple_response(response_text: str) -> str:
    """Process the simple agent response for display"""
    return response_text if response_text else "No response generated."

def submit_question():
    """Handle form submission"""
    # The processing state will be set when we actually start processing
    pass

def ask_simple_agent(question: str, conversation_history: list, conversation_id: Optional[str] = None) -> tuple[str, str]:
    """Ask the simple agent a question and return the response with conversation_id"""
    try:
        # Get the simple agent
        agent = get_simple_agent()
        
        # Generate response with conversation history
        result = agent.predict_with_history(
            new_message=question,
            conversation_history=conversation_history,
            conversation_id=conversation_id
        )
        
        return result["response"], result["conversation_id"]
        
    except Exception as e:
        logger.error(f"Error in ask_simple_agent: {str(e)}")
        return f"Error: {str(e)}", conversation_id or str(uuid.uuid4())

def main():
    # Initialize session state
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Streamlit UI
    st.title("Simple LLM Agent Chatbot")
    st.caption("Database-free version with in-memory conversation state")

    # Display current conversation info
    if st.session_state.conversation_id:
        st.sidebar.info(f"Conversation ID: {st.session_state.conversation_id[:8]}...")
        st.sidebar.info(f"Messages: {len(st.session_state.chat_history)}")
    else:
        st.sidebar.info("New conversation")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar with examples
    st.sidebar.title("Examples")
    examples = [
        "What is machine learning?",
        "Explain neural networks simply",
        "How does natural language processing work?",
        "What are the benefits of AI?",
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
                logger.info(f"Current conversation_id: {st.session_state.conversation_id}")
                
                # Ask the simple agent (pass current conversation history)
                response, new_conversation_id = ask_simple_agent(
                    user_input, 
                    st.session_state.chat_history[:-1],  # Exclude the just-added user message
                    st.session_state.conversation_id
                )
                
                # Update conversation_id
                st.session_state.conversation_id = new_conversation_id
                logger.info(f"New conversation_id: {st.session_state.conversation_id}")
                
                # Process and display response
                formatted_response = process_simple_response(response)
                message_placeholder.markdown(formatted_response)
                
                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
                
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
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.conversation_id = None
        st.session_state.processing = False
        st.rerun()
    
    # New conversation button
    if st.sidebar.button("🆕 New Conversation"):
        st.session_state.conversation_id = None
        st.sidebar.success("New conversation will start with next message")
    
    # Display agent info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Agent Info")
    st.sidebar.text("• Simple in-memory state")
    st.sidebar.text("• No database required")
    st.sidebar.text("• Session-based conversations")
    st.sidebar.text("• GPT foundation model (databricks-gpt-oss-20b)")
    
    # Show conversation history summary
    if st.session_state.chat_history:
        st.sidebar.markdown("### Conversation Stats")
        user_msgs = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        assistant_msgs = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
        st.sidebar.text(f"User messages: {user_msgs}")
        st.sidebar.text(f"Assistant messages: {assistant_msgs}")

if __name__ == "__main__":
    main()
