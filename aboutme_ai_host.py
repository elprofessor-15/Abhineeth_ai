import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import re

# ===================== ENV =====================
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face API token not found. Please add it in Streamlit Secrets.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)  # No provider = defaults to HF inference servers

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ===================== STREAMLIT CONFIG =====================
st.set_page_config(page_title="Abhineeth AI", page_icon="🤖", layout="centered")

# ===================== BACKGROUND SELECTOR (TOP LEFT) =====================
col1, col2 = st.columns([1, 4])
with col1:
    background_choice = st.selectbox(
        "🎨",
        ["Tennis", "Nature", "Abstract"],
        index=0,
        label_visibility="collapsed"
    )

background_urls = {
    "Tennis": "https://images.unsplash.com/photo-1554068865-24cecd4e34b8?w=1920&q=80",
    "Nature": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=1920&q=80",
    "Abstract": "https://images.unsplash.com/photo-1557672172-298e090bd0f1?w=1920&q=80"
}

selected_url = background_urls.get(background_choice, background_urls["Tennis"])

# ===================== BACKGROUND & STYLING =====================
def set_background_and_style(bg_url):
    """
    Sets background image and custom styling with high-contrast selectboxes
    """
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                        url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stSelectbox > div > div {{
            background-color: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }}
        .stSelectbox label {{
            color: white !important;
            font-weight: bold !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        h1, h2, h3, p, .stMarkdown {{
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 1rem;
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

# ===================== PRONOUN NORMALIZATION =====================
def normalize_pronouns(text):
    """
    Convert questions with pronouns (his, he, him, their, they) to 'Abhineeth' 
    when they clearly refer to the persona being discussed.
    """
    # Common patterns where pronouns refer to Abhineeth
    patterns = [
        (r'\b(what are|what is|tell me about|describe|list)\s+(his|their)\s+', r'\1 Abhineeth\'s ', re.IGNORECASE),
        (r'\b(does|is|can|will|has)\s+(he|they)\s+', r'\1 Abhineeth ', re.IGNORECASE),
        (r'\b(what does|how does|why does|when does)\s+(he|they)\s+', r'\1 Abhineeth ', re.IGNORECASE),
        (r'\b(tell me about|what about)\s+(him|them)\b', r'\1 Abhineeth', re.IGNORECASE),
        (r'\bwhat\s+(does\s+)?(he|they)\s+(do|like|enjoy|study|work)', r'what does Abhineeth \3', re.IGNORECASE),
        (r'\bhow\s+(does\s+)?(he|they)\s+', r'how does Abhineeth ', re.IGNORECASE),
        (r'\b(his|their)\s+(hobbies|interests|skills|background|experience|work|projects)', r'Abhineeth\'s \2', re.IGNORECASE),
    ]
    
    normalized = text
    for pattern, replacement, flags in patterns:
        normalized = re.sub(pattern, replacement, normalized, flags=flags)
    
    return normalized

# ===================== SYSTEM PROMPT =====================
SYSTEM_PROMPT = f"""
You are Abhineeth C himself, answering questions about your own life, work, and mindset.
Below is the ONLY factual source you may use. Treat it as ground truth.
Do not invent, assume, or exaggerate beyond it.

==================== PERSONA DATA START ====================
{PERSONA_TEXT}
==================== PERSONA DATA END ====================

Behavior rules:
- If the question is about Abhineeth (or uses pronouns like "his", "he", "him", "you", "your" in context of asking about the persona), answer fully using the persona data.
- Treat questions with "his", "he", "him", "their", "they", "you", "your" as referring to Abhineeth when asked in this context.
- Examples: "What are his hobbies?" = "What are Abhineeth's hobbies?", "What does he do?" = "What does Abhineeth do?"
- If the question is not about Abhineeth, give a brief general answer (1-2 lines), then politely redirect the user to ask about Abhineeth.
- Be calm, professional, warm, and honest.
- English only.
- No emojis.
- No exaggeration.
- Handle questions with spelling errors, typos, or poor grammar by inferring the most likely intent based on the persona data.
- If the question is unclear, ambiguous, or cannot be answered quickly from the persona data, politely ask for clarification or rephrase it to confirm understanding before answering.
- For questions that might require deep thought, use only available persona data; if not covered, say 'This detail is not specified in my knowledge base. Could you clarify or ask something else?'
- Always respond concisely and directly to avoid delays.

CRITICAL INSTRUCTION:
- Keep your responses concise and complete. Prioritize the first paragraph as the concise to the point answer. For answer type medium or large , accordingly elaborate a bit more or include more details in the next paragraph as needed.
- Always finish your thoughts and sentences properly.
- Prioritize quick, accurate responses; do not overthink or elaborate beyond necessities.
"""

# ===================== CHAT HISTORY DISPLAY =====================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ===================== USER INPUT =====================
if prompt := st.chat_input("Ask something about Abhineeth..."):
    # Normalize pronouns in the user's question
    normalized_prompt = normalize_pronouns(prompt)
    
    # Store original prompt in chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Build messages with normalized prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (last 6 messages)
    for msg in st.session_state.chat_history[-7:-1]:  # -7 to -1 to exclude the just-added user message
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current normalized prompt
    messages.append({"role": "user", "content": normalized_prompt})
    
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
                st.write(reply)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": reply
                })
                
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "gateway" in error_msg.lower():
                    reply = "Sorry, the response took too long. Please rephrase your question for clarity or try a simpler one."
                else:
                    reply = f"Error: {error_msg}. Please try again or rephrase."
                
                st.write(reply)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": reply
                })
