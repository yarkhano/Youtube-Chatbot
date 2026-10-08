import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CACHE_TTL, MAX_CACHED_VIDEOS
from models import get_embeddings
from youtube_service import fetch_transcript


@st.cache_resource(
    show_spinner=False,
    ttl=CACHE_TTL,
    max_entries=MAX_CACHED_VIDEOS,
)
def process_video(
    video_id,
    embedding_model,
    chunk_size,
    chunk_overlap,
):
    text = fetch_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = splitter.create_documents(
        [text],
        metadatas=[{"video_id": video_id}],
    )

    embeddings = get_embeddings(embedding_model)

    return FAISS.from_documents(
        documents,
        embeddings,
    )