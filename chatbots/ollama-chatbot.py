import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama  # Replace with langchain_ollama.OllamaLLM if you upgrade
from dotenv import load_dotenv
import os


"""Module for creating a Streamlit chatbot interface using LangChain and Ollama's LLM."""

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

# Define the chat prompt template with system and user messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

st.title('Langchain Demo With LLAMA2 API')
input_text = st.text_input("Search the topic you want")

# Initialize Ollama LLM with the specified model
llm = Ollama(model="gemma3:1b")  
output_parser = StrOutputParser()

# Create the processing chain: prompt -> LLM -> output parser
chain = prompt | llm | output_parser

if input_text:
    # Invoke the chain with user input and display the result
    result = chain.invoke({"question": input_text})
    st.write(result)