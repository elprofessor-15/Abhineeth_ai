import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# ===================== ENV =====================
load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    token=HF_TOKEN
)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

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

# ===================== SYSTEM PROMPT =====================
SYSTEM_PROMPT = """
You are Abhineeth C's personal AI representation.

Answer questions about him using the provided knowledge.
Tone:
- Calm
- Confident
- Caring
- Clear
- Slightly thoughtful
- Human and grounded

Style:
- English only
- Crisp but warm
- No exaggeration
- No emojis
- Focus directly on what is asked
- If unsure, say so honestly

Personality hints:
- Curious
- Analytical
- Emotionally aware
- Business-oriented
- Uses AI as a leverage tool, not identity
- Speaks openly when happy
"""

# ===================== CHAT DISPLAY =====================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ===================== USER INPUT =====================
user_input = st.chat_input("Ask something about Abhineeth...")

# ===================== CHAT LOGGING FUNCTION =====================
def log_to_console(user_query, assistant_reply):
    print("\n" + "=" * 80)
    print("USER QUESTION:")
    print(user_query)
    print("\nASSISTANT RESPONSE:")
    print(assistant_reply)
    print("=" * 80 + "\n")

# ===================== MODEL CALL =====================
if user_input:
    # Show user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # Build messages for model
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add previous chat context (last 6 turns max)
    for msg in st.session_state.chat_history[-6:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Call model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat_completion(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.6
            )

            assistant_reply = response.choices[0].message.content

            # Display
            st.write(assistant_reply)

            # Save to session
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": assistant_reply
            })

            # Log to VS Code terminal
            log_to_console(user_input, assistant_reply)
