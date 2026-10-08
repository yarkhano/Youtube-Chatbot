import os

import streamlit as st

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)
from rag import (
    build_chains,
    get_chat_history,
    prepare_search_query,
)
from vector_store import process_video
from youtube_service import get_youtube_id


st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="📺",
    layout="centered",
)

st.title("📺 YouTube Video Chat Assistant")
st.markdown(
    "Paste a YouTube link in the sidebar and chat with "
    "the video's content using Gemini."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None


with st.sidebar:
    st.header("Configuration")

    youtube_url = st.text_input(
        "Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    use_followup_context = st.checkbox(
        "Use conversation context for follow-up questions",
        value=True,
        help=(
            "Turn this off to skip the extra question-rewriting API call. "
            "When off, ask complete, standalone questions."
        ),
    )

    if st.button("Clear Conversation"):
        st.session_state.messages = []


if not youtube_url.strip():
    st.info(
        "👈 Please enter a YouTube video URL in the sidebar "
        "to get started."
    )
    st.stop()


video_id = get_youtube_id(youtube_url)

if not video_id:
    st.error("Invalid YouTube URL. Please enter a valid link.")
    st.stop()


if st.session_state.current_video_id != video_id:
    st.session_state.current_video_id = video_id
    st.session_state.messages = []


if not (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
):
    st.error(
        "Add GOOGLE_API_KEY to your .env file, then restart the app."
    )
    st.stop()


try:
    with st.spinner("Loading video transcript and index..."):
        vector_store = process_video(
            video_id=video_id,
            embedding_model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

except Exception as exc:
    st.error(
        "Could not process the video. It needs accessible English "
        "captions. YouTube restrictions, network issues, or "
        "embedding errors may also cause this."
    )
    st.code(str(exc), language=None)
    st.stop()


try:
    rewrite_chain, answer_chain = build_chains(vector_store)

except Exception as exc:
    st.error(f"Could not initialize the assistant: {exc}")
    st.stop()


st.success("Video successfully loaded! Ask away below.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_query := st.chat_input(
    "What would you like to know about this video?"
):
    user_query = user_query.strip()

    if not user_query:
        st.stop()

    history = get_chat_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Preparing your question..."):
                search_query = prepare_search_query(
                    question=user_query,
                    history=history,
                    rewrite_chain=rewrite_chain,
                    use_followup_context=use_followup_context,
                )

            answer = st.write_stream(
                answer_chain.stream(
                    {
                        "search_query": search_query,
                        "question": user_query,
                        "history": history,
                    }
                )
            )

            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(
                    "Gemini returned an empty text response."
                )

    except Exception as exc:
        st.error(
            f"Could not generate an answer. Please try again. "
            f"Details: {exc}"
        )

    else:
        st.session_state.messages.extend(
            [
                {
                    "role": "user",
                    "content": user_query,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]
        )