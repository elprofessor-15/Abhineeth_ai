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

client = InferenceClient(token=HF_TOKEN)

# ✅ Router-safe chat model (works on Streamlit Cloud)
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

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
length_map = {
    "Small": 80,
    "Medium": 180,
    "Large": 350
}

response_size = st.selectbox(
    "Response length",
    ["Small", "Medium", "Large"],
    index=1
)

max_tokens = length_map[response_size]

# ===================== LOAD PERSONA =====================
with open("persona.txt", "r", encoding="utf-8") as f:
    PERSONA_TEXT = f.read()

# ===================== SYSTEM PROMPT (FIXED) =====================
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
            response = client.chat_completion(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.6
            )

            assistant_reply = response.choices[0].message.content.strip()

            st.write(assistant_reply)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_reply
            })

            log_to_console(user_input, assistant_reply)
