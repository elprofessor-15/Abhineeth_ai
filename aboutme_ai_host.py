import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# ===================== ENV =====================
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face API token not found. Please add it in Streamlit Secrets.")
    st.stop()

client = InferenceClient(
    provider="sambanova",
    token=HF_TOKEN
)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ===================== STREAMLIT CONFIG =====================
st.set_page_config(page_title="Abhineeth AI", page_icon="🤖", layout="centered")

# ===================== BACKGROUND SELECTOR (TOP LEFT) =====================
col1, col2 = st.columns([1, 4])

with col1:
    background_choice = st.selectbox(
        "🎨",
        ["Wimbledon Centre Court", "Nature/Forest", "Abstract Gradient"],
        index=0,
        label_visibility="collapsed"
    )

background_urls = {
    "Wimbledon Centre Court": "https://images.unsplash.com/photo-1554068865-24cecd4e34b8?w=1920&q=80",
    "Nature/Forest": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1920&q=80",
    "Abstract Gradient": "https://images.unsplash.com/photo-1557672172-298e090bd0f1?w=1920&q=80"
}

selected_url = background_urls.get(background_choice, background_urls["Wimbledon Centre Court"])

# ===================== BACKGROUND & STYLING =====================
def set_background_and_style(bg_url):
    """
    Sets background image and custom styling with high-contrast selectboxes
    """
    st.markdown(f"""
    <style>
    /* Background Image */
    .stApp {{
        background-image: url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Dark overlay */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.65);
        z-index: 0;
        pointer-events: none;
    }}

    .main > div {{
        position: relative;
        z-index: 1;
    }}

    /* Title */
    h1 {{
        color: #ffffff !important;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.95) !important;
        font-weight: bold !important;
        background: rgba(0, 0, 0, 0.35);
        padding: 15px;
        border-radius: 10px;
    }}

    /* Caption */
    .stCaption {{
        color: #f0f0f0 !important;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.9) !important;
        background: rgba(0,0,0,0.45);
        padding: 10px;
        border-radius: 8px;
        font-size: 15px !important;
    }}

    /* ──────────────────────────────────────────────
       SELECTBOX - HIGH CONTRAST BLACK BACKGROUND
    ────────────────────────────────────────────── */
    /* Label */
    .stSelectbox label {{
        color: #ffffff !important;
        text-shadow: 2px 2px 6px rgba(0,0,0,1) !important;
        font-weight: bold !important;
        font-size: 15px !important;
    }}

    /* Main container */
    div[data-baseweb="select"] {{
        background-color: #000000 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.55) !important;
    }}

    /* Selected value area → solid black + white text */
    div[data-baseweb="select"] > div {{
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        text-shadow: 1px 1px 5px rgba(0,0,0,1) !important;
        background: #000000 !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        min-height: 38px;
        display: flex;
        align-items: center;
    }}

    /* Arrow icon */
    div[data-baseweb="select"] svg {{
        fill: #ffffff !important;
    }}

    /* Dropdown menu when opened */
    [data-baseweb="popover"] {{
        background-color: #0f0f0f !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }}

    [role="listbox"] {{
        background-color: #0f0f0f !important;
    }}

    [role="option"] {{
        color: #e8e8e8 !important;
        background-color: #0f0f0f !important;
        padding: 8px 12px !important;
    }}

    [role="option"]:hover {{
        background-color: #1a5bb8 !important;
        color: #ffffff !important;
    }}

    /* Chat bubbles */
    .stChatMessage[data-testid="user-message"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 18px 18px 5px 18px !important;
        padding: 16px 22px !important;
        margin: 12px 0 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }}

    .stChatMessage[data-testid="assistant-message"] {{
        background: rgba(255, 255, 255, 0.98) !important;
        border-radius: 18px 18px 18px 5px !important;
        padding: 16px 22px !important;
        margin: 12px 0 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
        border-left: 5px solid #4CAF50;
        border: 2px solid rgba(0, 0, 0, 0.1);
    }}

    /* Chat input */
    .stChatInputContainer {{
        background: rgba(255, 255, 255, 0.97) !important;
        border-radius: 25px !important;
        padding: 8px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(0,0,0,0.15);
    }}

    /* Others */
    hr {{
        border-color: rgba(255, 255, 255, 0.25) !important;
        margin: 20px 0 !important;
    }}

    .stSpinner > div {{
        border-color: #4CAF50 transparent transparent transparent !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Apply styling
set_background_and_style(selected_url)

# ===================== MAIN CONTENT =====================
st.title("Ask Anything About Abhineeth")

st.caption(
    "This AI answers professional questions about Abhineeth — research, career, skills, mindset, and work. "
    "For personal matters, please reach out to Abhineeth directly."
)

st.markdown("---")

# ===================== SESSION STATE =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===================== RESPONSE LENGTH =====================
length_map = {
    "Small": 150,
    "Medium": 300,
    "Large": 500
}

response_size = st.selectbox(
    "Response length",
    ["Small", "Medium", "Large"],
    index=1
)

max_tokens = length_map[response_size]

# ===================== LOAD PERSONA =====================
try:
    with open("persona.txt", "r", encoding="utf-8") as f:
        PERSONA_TEXT = f.read()
except FileNotFoundError:
    st.error("persona.txt file not found. Please ensure it exists in the app directory.")
    st.stop()

# ===================== SYSTEM PROMPT =====================
SYSTEM_PROMPT = f"""
You are Abhineeth C himself, answering questions about your own life, work, and mindset.

Below is the ONLY factual source you may use. Treat it as ground truth.
Do not invent, assume, or exaggerate beyond it.

====================
PERSONA DATA START
====================
{PERSONA_TEXT}
====================
PERSONA DATA END
====================

Behavior rules:
- If the question is about Abhineeth, answer fully using the persona data
- If the question is not about Abhineeth, give a brief general answer (1–2 lines),
  then politely redirect the user to ask about Abhineeth
- Be calm, professional, warm, and honest
- English only
- No emojis
- No exaggeration

CRITICAL INSTRUCTION:
- Keep your responses concise and complete
- Always finish your thoughts and sentences properly
"""

# ===================== CHAT HISTORY DISPLAY =====================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ===================== USER INPUT =====================
if prompt := st.chat_input("Ask something about Abhineeth..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.6,
                    top_p=0.9,
                    stop=["\n\n\n", "User:", "Question:"]
                )

                reply = response.choices[0].message.content.strip()

                # Optional: your previous ensure_complete_response logic here if needed
                st.write(reply)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": reply
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
