import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama  # Replace with langchain_ollama.OllamaLLM if you upgrade
from dotenv import load_dotenv
import os

"""Streamlit application for interacting with LLAMA2 API through LangChain."""

load_dotenv()

# Configure LangChain environment variables
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

def create_chat_prompt():
    """Create and return a ChatPromptTemplate with system and user message templates."""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ])

# Initialize Streamlit UI components
st.title('Langchain Demo With LLAMA2 API')
input_text = st.text_input("Search the topic you want")

def initialize_llm_chain():
    """
    Initialize and return the LLM processing chain.
    
    Returns:
        A processing chain consisting of: prompt -> LLM -> output parser
    """
    llm = Ollama(model="gemma3:1b")  # Consider upgrading to langchain_ollama.OllamaLLM
    output_parser = StrOutputParser()
    return create_chat_prompt() | llm | output_parser

# Main processing logic
if input_text:
    chain = initialize_llm_chain()
    result = chain.invoke({"question": input_text})
    st.write(result)