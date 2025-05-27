import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama  # Replace with langchain_ollama.OllamaLLM if you upgrade
# from dotenv import load_dotenv
# load_dotenv()

# Define prompt using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Streamlit app UI
st.title('Langchain Demo With LLAMA2 API')
input_text = st.text_input("Search the topic you want")

# Initialize LLM and parser
llm = Ollama(model="gemma3")  # Consider upgrading to langchain_ollama.OllamaLLM
output_parser = StrOutputParser()

# Define chain
chain = prompt | llm | output_parser

# Run the chain on input
if input_text:
    result = chain.invoke({"question": input_text})
    st.write(result)
