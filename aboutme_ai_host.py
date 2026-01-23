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

# ✅ Using Inference Providers - the NEW way to access models on HF
client = InferenceClient(
    provider="sambanova",  # Fast, free-tier friendly provider
    token=HF_TOKEN
)

# ✅ Model available through SambaNova provider
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ===================== STREAMLIT CONFIG =====================
st.set_page_config(page_title="Abhineeth AI", page_icon="🤖", layout="centered")
st.title("Ask Anything About Abhineeth")

st.caption(
    "This AI answers professional questions about Abhineeth — research, career, skills, mindset, and work. "
    "For personal matters, please reach out to Abhineeth directly."
)

# ===================== SESSION STATE =====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===================== RESPONSE LENGTH CONTROL =====================
# Increased token limits to allow for complete responses
length_map = {
    "Small": 150,    # Increased from 80
    "Medium": 300,   # Increased from 180
    "Large": 500     # Increased from 350
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
    
    # If text ends with proper punctuation, it's likely complete
    if text and text[-1] in '.!?':
        return text
    
    # Find the last complete sentence
    last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    
    if last_period > len(text) * 0.5:  # If we have at least 50% of the text with complete sentences
        return text[:last_period + 1].strip()
    
    # If no good breaking point, try to end at last complete word
    if text:
        # Add ellipsis to indicate truncation
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

    # Build messages for the API using chat.completions format
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
                # Use the new chat.completions interface with enhanced parameters
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.6,
                    top_p=0.9,  # Helps with response quality
                    stop=["\n\n\n", "User:", "Question:"]  # Stop sequences to prevent rambling
                )

                assistant_reply = response.choices[0].message.content.strip()
                
                # Check if response was cut off (finish_reason will be 'length' if truncated)
                finish_reason = response.choices[0].finish_reason
                
                if finish_reason == "length":
                    # Response was truncated, ensure it ends properly
                    assistant_reply = ensure_complete_response(assistant_reply)
                    
                    # Optional: Add a subtle indicator that response was condensed
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
                
                # Helpful debugging info
                if "model_not_supported" in error_msg:
                    st.info("""
                    **Model not available.** Try one of these alternatives:
                    - meta-llama/Llama-3.1-8B-Instruct
                    - meta-llama/Llama-3.3-70B-Instruct
                    - Qwen/Qwen2.5-72B-Instruct
                    
                    Update MODEL_NAME in the code to one of these models.
                    """)
                elif "provider" in error_msg.lower():
                    st.info("""
                    **Provider issue.** Try changing the provider to:
                    - "together" (Together AI)
                    - "replicate" (Replicate)
                    - "hf-inference" (Hugging Face native)
                    """)
                
                print(f"ERROR: {e}")