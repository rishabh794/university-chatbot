import streamlit as st
import joblib
import json
import random
import time

# --- NLTK & Preprocessing Block ---
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data (if not already downloaded by the training script)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK data for app...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define the preprocessing function *exactly* as in train_model.py
def preprocess_text(text):
    """
    Tokenizes and lemmatizes the input text.
    """
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens
# --- End of NLTK Block ---


# --- 1. Load Trained Model and Data ---
try:
    pipeline = joblib.load('chatbot_model.joblib')
    with open('intents.json', 'r') as f:
        data = json.load(f)
    print("Model and intents loaded successfully.")
except Exception as e:
    st.error(f"Error loading models or data: {e}. Please run train_model.py first.")
    st.stop()

# Create a dictionary of responses for fast lookup
responses = {intent['tag']: intent['responses'] for intent in data['intents']}

# --- 2. Create the Chatbot Logic ---
def get_response(user_input):
    """
    Get a response from the chatbot model.
    Returns the response and the predicted tag.
    """
    predicted_tag = pipeline.predict([user_input])[0]
    
    try:
        response_list = responses[predicted_tag]
        bot_response = random.choice(response_list)
    except KeyError:
        # Fallback response
        bot_response = "I'm not sure how to respond to that. Can you try rephrasing?"
        
    return bot_response, predicted_tag

# --- 3. New Function for Suggested Questions ---
def handle_suggestion(question):
    """
    Called when a suggested question button is clicked.
    Adds the question and the bot's response to the chat history.
    """
    # 1. Add user's suggested question to chat
    st.session_state.messages.append({"role": "user", "content": question})
    
    # 2. Get and add bot's response
    bot_response, tag = get_response(question)
    st.session_state.messages.append({"role": "assistant", "content": bot_response, "tag": tag})

# --- 4. Build the Streamlit UI ---
st.set_page_config(page_title="University Chatbot", layout="wide")
st.title("🎓 University AI Assistant")
st.caption("Ask me about admissions, courses, fees, or contact info.")

# --- 5. NEW: Sidebar ---
with st.sidebar:
    st.header("About")
    st.info(
        "This is an intent-based chatbot for a B.Tech university. "
        "It uses a `LinearSVC` model trained on custom intents to answer questions. "
        "This project fulfills the requirements for the AI Project Submission."
    )
    st.subheader("Model")
    st.markdown("`scikit-learn`: `LinearSVC` + `TfidfVectorizer`")
    st.subheader("Frontend")
    st.markdown("`Streamlit`")
    st.subheader("Dataset")
    st.markdown("`intents.json` (11 intents, 100 patterns)")


# --- 6. Initialize and Display Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If it's an assistant message and has a tag, show the debug caption
        if message["role"] == "assistant" and "tag" in message:
            st.caption(f"Debug: Predicted intent = '{message['tag']}'")

# --- 7. NEW: Suggested Questions (Only show if chat is empty) ---
if not st.session_state.messages:
    st.markdown("---")
    st.subheader("Or, try one of these suggestions:")
    
    # Use columns for a neater layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button(
            "What courses do you offer?", 
            on_click=handle_suggestion, 
            args=["What courses do you offer?"],
            use_container_width=True
        )
        st.button(
            "What are the fees?", 
            on_click=handle_suggestion, 
            args=["What are the fees?"],
            use_container_width=True
        )
    with col2:
        st.button(
            "How do I apply for admissions?", 
            on_click=handle_suggestion, 
            args=["How do I apply for admissions?"],
            use_container_width=True
        )
        st.button(
            "Tell me about hostel facilities", 
            on_click=handle_suggestion, 
            args=["Tell me about hostel facilities"],
            use_container_width=True
        )
    with col3:
        st.button(
            "When is the application deadline?", 
            on_click=handle_suggestion, 
            args=["When is the application deadline?"],
            use_container_width=True
        )
        st.button(
            "How do I contact the university?", 
            on_click=handle_suggestion, 
            args=["How do I contact the university?"],
            use_container_width=True
        )
    st.markdown("---")


# --- 8. Handle New User Input (from chat box) ---
if prompt := st.chat_input("What would you like to know?"):
    
    # 1. Add user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get and add bot's response
    # NEW: Use a "thinking" spinner for better UX
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(0.5) # Add a small, artificial delay
            bot_response, tag = get_response(prompt)
            st.markdown(bot_response)
            st.caption(f"Debug: Predicted intent = '{tag}'")
        
    # Add the bot's response (and tag) to the session state
    st.session_state.messages.append({"role": "assistant", "content": bot_response, "tag": tag})