import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama  # Replace with langchain_ollama.OllamaLLM if you upgrade
from dotenv import load_dotenv
import os


load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGSMITH_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

st.title('Langchain Demo With LLAMA2 API')
input_text = st.text_input("Search the topic you want")

llm = Ollama(model="gemma3:1b")  
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    result = chain.invoke({"question": input_text})
    st.write(result)