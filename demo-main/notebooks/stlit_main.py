import os
import streamlit as st
import asyncio
import json
from typing import Dict, Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import GenieAPI
import requests

# Configuration
space_id = os.getenv("GENIE_SPACE_ID")
token = os.getenv("DATABRICKS_TOKEN")
host = os.environ.get('DATABRICKS_HOST'),
workspace_client = WorkspaceClient(
    host=os.environ.get('DATABRICKS_HOST'),
    client_id=os.environ.get('DATABRICKS_CLIENT_ID'),
    client_secret=os.environ.get('DATABRICKS_CLIENT_SECRET'),
    auth_type="pat",
    token=token
)
genie_api = GenieAPI(workspace_client.api_client)
#conversation_id = None

def get_query_results(conversation_id, message_id):
    hostname = host[0]
    url = f"https://{hostname}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result"
    response = requests.get(url, headers={'Authorization': f'Bearer {token}'})
    return response.json()

# Logging setup
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper functions (ask_genie and process_query_results remain the same)
async def ask_genie(question: str, space_id: str, conversation_id: Optional[str] = None) -> tuple[str, str]:
    try:
        loop = asyncio.get_running_loop()
        if conversation_id is None:
            initial_message = await loop.run_in_executor(None, genie_api.start_conversation_and_wait, space_id, question)
            conversation_id = initial_message.conversation_id
        else:
            initial_message = await loop.run_in_executor(None, genie_api.create_message_and_wait, space_id, conversation_id, question)

        query_result = None
        if initial_message.query_result is not None:
            query_result = await loop.run_in_executor(None, genie_api.get_message_query_result,
                space_id, initial_message.conversation_id, initial_message.id)

        message_content = await loop.run_in_executor(None, genie_api.get_message,
            space_id, initial_message.conversation_id, initial_message.id)

        if query_result and query_result.statement_response:
            results = await loop.run_in_executor(None, get_query_results,
                initial_message.conversation_id, initial_message.id)
            logger.info(f"results: {str(results)}")
            
            query_description = ""
            for attachment in message_content.attachments:
                if attachment.query and attachment.query.description:
                    query_description = attachment.query.description
                    break

            return json.dumps({
                "columns": results['statement_response']['manifest']['schema']['columns'],
                "data": results['statement_response']['result']['data_typed_array'],
                "query_description": query_description
            }), conversation_id

        if message_content.attachments:
            for attachment in message_content.attachments:
                if attachment.text and attachment.text.content:
                    return json.dumps({"message": attachment.text.content}), conversation_id

        return json.dumps({"message": message_content.content}), conversation_id
    except Exception as e:
        logger.error(f"Error in ask_genie: {str(e)}")
        return json.dumps({"error": "An error occurred while processing your request."}), conversation_id

def process_query_results(answer_json: Dict) -> str:
    response = ""
    logger.debug(f"answer_json: {str(answer_json)}")
    if "query_description" in answer_json and answer_json["query_description"]:
        response += f"## Descrição da Consulta\n\n{answer_json['query_description']}\n\n"

    if "columns" in answer_json and "data" in answer_json:
        response += "## Resultados da Consulta\n\n"
        columns = answer_json["columns"]
        data = answer_json["data"]
        if len(data) > 0: 
            header = "| " + " | ".join(col['name'] for col in columns) + " |"
            separator = "|" + "|".join(["---" for _ in columns]) + "|"
                
            response += header + "\n" + separator + "\n"
                
            for row in data:
                formatted_row = []
                for value, col_schema in zip(row['values'], columns):
                    if value is None or value.get('str') is None:
                        formatted_value = "NULL"
                    elif col_schema['type_name'] in ['DECIMAL', 'DOUBLE', 'FLOAT']:
                        formatted_value = f"{float(value['str']):,.2f}"
                    elif col_schema['type_name'] in ['INT', 'BIGINT']:
                        formatted_value = f"{int(value['str']):,}"
                    else:
                        formatted_value = value['str']
                    formatted_row.append(formatted_value)
                response += "| " + " | ".join(formatted_row) + " |\n"
        else:
            response += f"Unexpected column format: {columns}\n\n"
    elif "message" in answer_json:
        response += f"{answer_json['message']}\n\n"
    else:
        response += "No data available.\n\n"
    
    return response

def submit_question():
    st.session_state.processing = True


def main():

    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None

    # Streamlit UI
    st.title("Chatbot powered by Genie")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar with examples
    st.sidebar.title("Examples")
    examples = [
        "Descreva os Datasets",
        "Qual é o preço médio dos produtos por categoria?",
        "Quantos produtos existem em cada categoria?",
    ]

    # User input
    user_input = st.chat_input("Digite sua mensagem aqui...", on_submit=submit_question, disabled=st.session_state.processing)

    for example in examples:
        if st.sidebar.button(example):
            st.session_state.prompt = example
            user_input = example


    if user_input:# and not st.session_state.processing:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get bot response
        with st.chat_message("assistant"):
            st.session_state.processing = True

            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            try:
                logger.info(f"Conversation_id: {str(st.session_state.conversation_id)}")
                answer, new_conversation_id = asyncio.run(ask_genie(user_input, space_id, st.session_state.conversation_id))
                st.session_state.conversation_id = new_conversation_id
                
                logger.info(f"New conversation_id: {str(st.session_state.conversation_id)}")
                answer_json = json.loads(answer)
                response = process_query_results(answer_json)
                message_placeholder.markdown(response)
                # Add bot response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.session_state.processing = False
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                message_placeholder.markdown("An error occurred while processing your request.")
                st.session_state.processing = False
            st.rerun()


    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.processing = False
        st.rerun()

if __name__ == "__main__":
    main()