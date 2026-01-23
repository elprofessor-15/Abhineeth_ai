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

# ===================== BACKGROUND & STYLING =====================
def set_background_and_style():
    """
    Sets background image and custom chat styling
    """
    st.markdown("""
    <style>
    /* Background Image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1554068865-24cecd4e34b8?w=1920");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Dark overlay for readability */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 0;
        pointer-events: none;
    }
    
    /* Make content appear above overlay */
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    /* Title styling */
    h1 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        font-weight: bold !important;
    }
    
    /* Caption styling */
    .stCaption {
        color: #e0e0e0 !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    }
    
    /* User message bubble (right side) */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 18px 18px 5px 18px !important;
        padding: 15px 20px !important;
        margin: 10px 0 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
    }
    
    .stChatMessage[data-testid="user-message"] p {
        color: white !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    
    /* Assistant message bubble (left side) */
    .stChatMessage[data-testid="assistant-message"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 18px 18px 18px 5px !important;
        padding: 15px 20px !important;
        margin: 10px 0 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
        border-left: 4px solid #4CAF50;
    }
    
    .stChatMessage[data-testid="assistant-message"] p {
        color: #1a1a1a !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    
    /* Chat input box */
    .stChatInputContainer {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 25px !important;
        padding: 5px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Selectbox styling */
    .stSelectbox {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
        padding: 5px;
    }
    
    .stSelectbox label {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
        font-weight: bold !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #4CAF50 transparent transparent transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply styling
set_background_and_style()

# ===================== STREAMLIT CONFIG =====================
st.set_page_config(page_title="Abhineeth AI", page_icon="🤖", layout="centered")
st.title("Ask Anything About Abhineeth")

st.caption(
    "This AI answers professional questions about Abhineeth — research, career, skills, mindset, and work. "
    "For personal matters, please reach out to Abhineeth directly."
)

# ===================== BACKGROUND IMAGE SELECTOR (OPTIONAL) =====================
st.markdown("---")
with st.expander("🎨 Change Background Image"):
    background_choice = st.radio(
        "Choose a background:",
        ["Tennis Court (Default)", "Nature/Forest", "Abstract Gradient", "Stadium", "Custom URL"],
        index=0
    )
    
    background_urls = {
        "Tennis Court (Default)": "https://images.unsplash.com/photo-1554068865-24cecd4e34b8?w=1920",
        "Nature/Forest": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1920",
        "Abstract Gradient": "https://images.unsplash.com/photo-1557672172-298e090bd0f1?w=1920",
        "Stadium": "https://images.unsplash.com/photo-1508098682722-e99c43a406b2?w=1920"
    }
    
    if background_choice == "Custom URL":
        custom_url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        if custom_url:
            st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("{custom_url}") !important;
            }}
            </style>
            """, unsafe_allow_html=True)
    else:
        selected_url = background_urls.get(background_choice)
        if selected_url:
            st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("{selected_url}") !important;
            }}
            </style>
            """, unsafe_allow_html=True)

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
