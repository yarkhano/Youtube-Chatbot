import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource(show_spinner=False)
def get_embeddings(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource(show_spinner=False)
def get_llm(model_name):
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_retries=2,
    )