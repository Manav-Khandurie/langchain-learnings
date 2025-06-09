from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

"""Main module for Langchain demo with DeepSeek Chat integration.
Handles LLM initialization, prompt setup, and Streamlit UI configuration.
"""

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

def initialize_deepseek_llm():
    """Initialize and return a ChatOpenAI instance configured for DeepSeek's API.
    
    Returns:
        ChatOpenAI: An instance configured with DeepSeek's model and API endpoint.
    """
    return ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",  # DeepSeek API endpoint
        api_key=os.getenv("DEEPSEEK_API_KEY")    # API key from environment variables
    )

llm = initialize_deepseek_llm()

def create_chat_prompt_template():
    """Create and return a predefined chat prompt template.
    
    Returns:
        ChatPromptTemplate: A template with system and user message placeholders.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user's queries."),
            ("user", "Question: {question}")
        ]
    )

prompt = create_chat_prompt_template()

def setup_streamlit_ui():
    """Set up and run the Streamlit UI for the Langchain demo.
    
    Creates a text input field, processes user queries through the LLM chain,
    and displays responses.
    """
    st.title('Langchain Demo With DeepSeek Chat')
    input_text = st.text_input("Search the topic you want")

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser  # Create processing pipeline

    if input_text:
        response = chain.invoke({'question': input_text})
        st.write(response)

if __name__ == "__main__":
    setup_streamlit_ui()