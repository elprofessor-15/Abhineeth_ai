import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import base64

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
    Sets background image and custom chat styling with enhanced text visibility
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
    
    /* Darker overlay for better text readability */
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
    
    /* Make content appear above overlay */
    .main > div {{
        position: relative;
        z-index: 1;
    }}
    
    /* Title styling - Enhanced visibility */
    h1 {{
        color: #ffffff !important;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.95), 0 0 10px rgba(0, 0, 0, 0.8) !important;
        font-weight: bold !important;
        background: rgba(0, 0, 0, 0.3);
        padding: 15px;
        border-radius: 10px;
    }}
    
    /* Caption styling - Enhanced visibility */
    .stCaption {{
        color: #f0f0f0 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.95), 0 0 8px rgba(0, 0, 0, 0.7) !important;
        background: rgba(0, 0, 0, 0.4);
        padding: 10px;
        border-radius: 8px;
        font-size: 15px !important;
    }}
    
    /* User message bubble - Enhanced contrast */
    .stChatMessage[data-testid="user-message"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 18px 18px 5px 18px !important;
        padding: 16px 22px !important;
        margin: 12px 0 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }}
    
    .stChatMessage[data-testid="user-message"] p {{
        color: #ffffff !important;
        font-size: 17px !important;
        line-height: 1.6 !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5) !important;
        font-weight: 500 !important;
    }}
    
    /* Assistant message bubble - Maximum readability */
    .stChatMessage[data-testid="assistant-message"] {{
        background: rgba(255, 255, 255, 0.98) !important;
        border-radius: 18px 18px 18px 5px !important;
        padding: 16px 22px !important;
        margin: 12px 0 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(10px);
        border-left: 5px solid #4CAF50;
        border: 2px solid rgba(0, 0, 0, 0.1);
    }}
    
    .stChatMessage[data-testid="assistant-message"] p {{
        color: #1a1a1a !important;
        font-size: 17px !important;
        line-height: 1.7 !important;
        font-weight: 500 !important;
    }}
    
    /* Chat input box - Enhanced visibility */
    .stChatInputContainer {{
        background: rgba(255, 255, 255, 0.98) !important;
        border-radius: 25px !important;
        padding: 8px !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 0, 0, 0.1);
    }}
    
    .stChatInputContainer input {{
        color: #1a1a1a !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }}
    
    /* CRITICAL FIX: Selectbox styling - WHITE BACKGROUND with DARK TEXT */
    .stSelectbox {{
        background: transparent !important;
    }}
    
    /* Label styling - white text with shadow */
    .stSelectbox label {{
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.95), 0 0 8px rgba(0, 0, 0, 0.7) !important;
        font-weight: bold !important;
        font-size: 15px !important;
    }}
    
    /* Dropdown box itself - WHITE BACKGROUND */
    .stSelectbox div[data-baseweb="select"] {{
        background-color: #ffffff !important;
        border: 2px solid rgba(0, 0, 0, 0.3) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }}
    
    /* Selected text in dropdown - DARK TEXT */
    .stSelectbox div[data-baseweb="select"] > div {{
        color: #1a1a1a !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }}
    
    /* Dropdown arrow icon */
    .stSelectbox svg {{
        fill: #1a1a1a !important;
    }}
    
    /* Dropdown menu when opened - WHITE BACKGROUND */
    [data-baseweb="popover"] {{
        background-color: #ffffff !important;
    }}
    
    [role="listbox"] {{
        background-color: #ffffff !important;
    }}
    
    /* Dropdown options - DARK TEXT */
    [role="option"] {{
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }}
    
    [role="option"]:hover {{
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }}
    
    /* Horizontal rule styling */
    hr {{
        border-color: rgba(255, 255, 255, 0.3) !important;
        margin: 20px 0 !important;
    }}
    
    /* Spinner styling */
    .stSpinner > div {{
        border-color: #4CAF50 transparent transparent transparent !important;
    }}
    
    /* Column styling for background selector */
    [data-testid="column"] {{
        background: transparent !important;
    }}
    
    /* Fix for the emoji selector icon */
    .stSelectbox:first-of-type div[data-baseweb="select"] {{
        min-height: 45px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Apply styling with selected background
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

# ===================== RESPONSE LENGTH CONTROL =====================
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

# ===================== ENHANCED SYSTEM PROMPT =====================
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
- If you're running low on space, prioritize completing your current sentence over starting new ones
- End naturally with proper punctuation
"""

# ===================== CHAT DISPLAY =====================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ===================== USER INPUT =====================
user_input = st.chat_input("Ask something about Abhineeth...")

# ===================== LOGGING =====================
def log_to_console(user_query, assistant_reply):
    print("\n" + "=" * 80)
    print("USER QUESTION:")
    print(user_query)
    print("\nASSISTANT RESPONSE:")
    print(assistant_reply)
    print("=" * 80 + "\n")

# ===================== RESPONSE COMPLETION HELPER =====================
def ensure_complete_response(text, max_length=None):
    """
    Ensures the response ends properly by trimming to last complete sentence.
    """
    text = text.strip()
    
    if text and text[-1] in '.!?':
        return text
    
    last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    
    if last_period > len(text) * 0.5:
        return text[:last_period + 1].strip()
    
    if text:
        return text.rsplit(' ', 1)[0].strip() + '...'
    
    return text

# ===================== MODEL CALL =====================
if user_input:
    # Save user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep last 6 messages as context
    for msg in st.session_state.chat_history[-6:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

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

                assistant_reply = response.choices[0].message.content.strip()
                finish_reason = response.choices[0].finish_reason
                
                if finish_reason == "length":
                    assistant_reply = ensure_complete_response(assistant_reply)
                    if not assistant_reply.endswith(('...', '.', '!', '?')):
                        assistant_reply += '.'

                st.write(assistant_reply)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_reply
                })

                log_to_console(user_input, assistant_reply)
            
            except Exception as e:
                error_msg = str(e)
                st.error(f"⚠️ Error: {error_msg}")
                
                if "model_not_supported" in error_msg:
                    st.info("""
                    **Model not available.** Try one of these alternatives:
                    - meta-llama/Llama-3.1-8B-Instruct
                    - meta-llama/Llama-3.3-70B-Instruct
                    - Qwen/Qwen2.5-72B-Instruct
                    """)
                elif "provider" in error_msg.lower():
                    st.info("""
                    **Provider issue.** Try changing the provider to:
                    - "together" (Together AI)
                    - "replicate" (Replicate)
                    - "hf-inference" (Hugging Face native)
                    """)
                
                print(f"ERROR: {e}")
