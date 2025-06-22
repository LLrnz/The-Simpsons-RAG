from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from rag import rag_chain
import os
import random
import base64


HOMER_AVATAR_URL = "assets/homer.png"
USER_AVATAR_EMOJI = "🍩"

SIMPSONS_SKY_BLUE = "#87CEEB"
SIMPSONS_YELLOW = "#FFD700"

ASSETS_DIR = "assets"
SOUND_FILES = []
FONT_FACE_CSS = ""

if os.path.isdir(ASSETS_DIR):
    try:
        SOUND_FILES = [
            os.path.join(ASSETS_DIR, f)
            for f in os.listdir(ASSETS_DIR)
            if os.path.isfile(os.path.join(ASSETS_DIR, f))
            and f.lower().endswith(".mp3")
        ]
    except Exception as e:
        st.toast(f"Could not load sound effects: {e}", icon="⚠️")
    try:
        font_filename = "Simpsonfont DEMO.otf"
        if os.path.isfile(os.path.join(ASSETS_DIR, font_filename)):
            font_path = os.path.join(ASSETS_DIR, font_filename)
            with open(font_path, "rb") as f:
                font_bytes = f.read()
            font_b64 = base64.b64encode(font_bytes).decode()

            font_extension = os.path.splitext(font_filename)[1].lower()
            font_format_map = {
                ".otf": "opentype",
            }
            font_format = font_format_map.get(font_extension)

            if font_format:
                FONT_FACE_CSS = f"""
                @font-face {{
                    font-family: 'SimpsonsFont';
                    src: url(data:font/{font_format};charset=utf-8;base64,{font_b64}) format('{font_format}');
                    font-weight: normal;
                    font-style: normal;
                }}
                """
            else:
                st.toast(f"Unsupported font format for {font_filename}", icon="⚠️")
        else:
            st.toast(
                f"Font file '{font_filename}' not found in '{ASSETS_DIR}'.", icon="⚠️"
            )
    except Exception as e:
        st.toast(f"Could not load custom font: {e}", icon="⚠️")


st.set_page_config(page_title="SimpsonGPT", page_icon="🍩", layout="wide")

st.markdown(
    f"""
<style>
    {FONT_FACE_CSS}
    body, 
    [data-testid="stAppViewContainer"], 
    .stTextInput input, 
    .stButton button, 
    .stChatInput textarea, 
    .stChatMessage p,
    .stMarkdown {{
        font-family: "Comic Sans MS", sans-serif !important;
    }}

    div[data-testid="stHeading"] h1 {{
        font-family: 'SimpsonsFont', "Comic Sans MS", sans-serif !important;
        color: {SIMPSONS_YELLOW};
        -webkit-text-stroke-width: 2px;
        -webkit-text-stroke-color: black;
        text-stroke-width: 2px;
        text-stroke-color: black;
        text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000, -2px 0 0 #000, 2px 0 0 #000, 0 -2px 0 #000, 0 2px 0 #000;
    }}
    div[data-testid="stHeading"] h1 p {{
        font-family: inherit !important; 
        color: inherit !important; 
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background-color: {SIMPSONS_SKY_BLUE};
    }}

    div[data-testid="stChatMessage"][aria-label^="Message from assistant"] img[data-testid="stChatMessageProfilePic"] {{
        object-fit: contain !important;
        border-radius: 8px !important;
        width: 60px !important;
        height: 60px !important;
        background-color: transparent !important;
    }}
    div[data-testid="stChatMessage"][aria-label^="Message from user"] span[data-testid="stChatMessageProfilePic"] {{
        font-size: 2rem;
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.title("HomerGPT")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Woo hoo! Let's talk about the Simpsons. What do you wanna know?",
        }
    ]


for message in st.session_state.messages:
    avatar_icon = (
        HOMER_AVATAR_URL if message["role"] == "assistant" else USER_AVATAR_EMOJI
    )
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about the Simpsons..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR_EMOJI):
        st.markdown(prompt)

    if SOUND_FILES:
        sound_to_play = random.choice(SOUND_FILES)
        try:
            with open(sound_to_play, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode()
            mime_type = "audio/mpeg"  # All sounds are mp3

            audio_html = f"""
                <audio autoplay="true">
                <source src="data:{mime_type};base64,{audio_b64}" type="{mime_type}">
                </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            print(f"Error playing sound {sound_to_play}: {e}")
            st.toast("D'oh! Couldn't play sound.", icon="🔇")

    with st.chat_message("assistant", avatar=HOMER_AVATAR_URL):
        with st.spinner("D'oh! Wiring brain..."):
            assistant_response_content = ""
            try:

                def stream_response_generator():
                    for chunk in rag_chain.stream(prompt):
                        yield chunk

                assistant_response_content = st.write_stream(
                    stream_response_generator()
                )

            except Exception as e:
                st.error(f"Ay caramba! There is an error: {e}")
                assistant_response_content = "Mmm... donuts... I mean, I got an error. Try asking something else."
                st.markdown(assistant_response_content)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response_content}
    )

