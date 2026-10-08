import re
from urllib.parse import parse_qs, urlparse

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

from config import CACHE_TTL, MAX_CACHED_VIDEOS


def get_youtube_id(url):
    url = url.strip()

    if not url:
        return None

    if "://" not in url:
        url = "https://" + url

    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path_parts = parsed.path.strip("/").split("/")
        video_id = None

        if host in {"youtu.be", "www.youtu.be"}:
            video_id = path_parts[0]

        elif host in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
            "youtube-nocookie.com",
            "www.youtube-nocookie.com",
        }:
            if parsed.path.rstrip("/") == "/watch":
                video_id = parse_qs(parsed.query).get("v", [None])[0]

            elif (
                len(path_parts) >= 2
                and path_parts[0] in {"shorts", "embed", "live", "v"}
            ):
                video_id = path_parts[1]

        if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            return video_id

    except ValueError:
        return None

    return None


@st.cache_data(
    show_spinner=False,
    ttl=CACHE_TTL,
    max_entries=MAX_CACHED_VIDEOS,
)
def fetch_transcript(video_id):
    transcript = YouTubeTranscriptApi().fetch(
        video_id,
        languages=["en"],
    )

    text = " ".join(item.text for item in transcript).strip()

    if not text:
        raise ValueError("The video transcript is empty.")

    return text