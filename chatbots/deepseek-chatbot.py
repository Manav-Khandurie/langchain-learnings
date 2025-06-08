from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith tracking configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

def initialize_deepseek_llm():
    """Initialize and return a DeepSeek ChatOpenAI instance.
    
    Returns:
        ChatOpenAI: Configured LLM instance with DeepSeek model settings.
    """
    return ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",  # DeepSeek API endpoint
        api_key=os.getenv("DEEPSEEK_API_KEY")    # API key from environment variables
    )

llm = initialize_deepseek_llm()

def create_chat_prompt_template():
    """Create and return a chat prompt template with system and user messages.
    
    Returns:
        ChatPromptTemplate: Configured prompt template for the chat interaction.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the user's queries."),
            ("user", "Question: {question}")
        ]
    )

prompt = create_chat_prompt_template()

def setup_streamlit_ui():
    """Set up the Streamlit user interface and handle chat interactions."""
    st.title('Langchain Demo With DeepSeek Chat')
    input_text = st.text_input("Search the topic you want")

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser  # Create a processing chain

    if input_text:
        response = chain.invoke({'question': input_text})  # Invoke the chain with user input
        st.write(response)  # Display the response in the Streamlit app

if __name__ == "__main__":
    setup_streamlit_ui()