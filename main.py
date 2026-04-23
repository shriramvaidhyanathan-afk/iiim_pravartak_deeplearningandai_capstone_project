import logging
import traceback

import streamlit as st
import os
import tempfile
from graph import ChatBot, ChatHistory
from redactor import TextRedactor
from traceabilitymanager import trace_manager, TraceabilityManager


# --- 1. CONFIGURATION & STYLES ---
def setup_page():
    st.set_page_config(page_title="PDF Assistant", page_icon="🤖", layout="wide")
    st.markdown("""
        <style>
            .main { background-color: #0e1117; }
            .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
            .stButton>button { background-color: #117aca; color: white; border-radius: 5px; width: 100%; }
            .status-box {
                padding: 10px; border-radius: 5px; background-color: #1e2630;
                border-left: 5px solid #117aca; margin-bottom: 20px;    
            }
        </style>
    """, unsafe_allow_html=True)


# --- 2. HELPERS ---
def sanitize_markdown(text: str) -> str:
    """Extracts content from markdown blocks if the LLM wraps the response."""
    if "```markdown" in text:
        text = text.split("```markdown")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()


@st.cache_resource
def get_chatbot(model_name):
    """Initializes the ChatBot once and caches it across reruns."""
    chat_bot = ChatBot(model_name=model_name)
    trace_manager.model_name = model_name
    return chat_bot.build_graph()


# --- 3. STATE MANAGEMENT ---
def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "feedback_scores" not in st.session_state:
        st.session_state.feedback_scores = {}
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = {
            "uploaded_pdf_paths": [],
            "current_status": "",
            "current_document_type": None,
            "has_atleast_one_pdf": False,
            "current_response": "",
            "current_request": "",
            "chat_history": ChatHistory(),
            "cancel_run": False
        }


def clear_session():
    st.session_state.messages = []
    st.session_state.processing = False
    st.session_state.chat_state = {
        "uploaded_pdf_paths": [],
        "current_status": "",
        "current_document_type": None,
        "has_atleast_one_pdf": False,
        "current_response": "",
        "current_request": "",
        "chat_history": ChatHistory(),
        "cancel_run": False
    }
    trace_manager.hard_reset()
    st.session_state.feedback_scores = {}
    st.rerun()


# --- 4. CORE LOGIC ---
def process_uploaded_files(uploaded_files):
    temp_paths = []
    # Create a single temporary directory for this batch
    temp_dir = tempfile.gettempdir()

    for uploaded_file in uploaded_files:
        # Join the temp directory path with the original file name
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        trace_manager.track_file(file_path)

        temp_paths.append(file_path)

    return temp_paths


def get_feedback_interaction_index(index):
    return index // 2


def run_pipeline(bot, prompt, temp_paths, status_placeholder):
    try:
        state = st.session_state.chat_state
        redactor = TextRedactor()
        state["current_request"] = redactor.redact(prompt)
        state["uploaded_pdf_paths"] = temp_paths

        response_state = bot.invoke(state)

        clean_md = sanitize_markdown(response_state["current_response"])

        # Display the result
        with st.chat_message("assistant"):
            st.markdown(clean_md)

        # Persist the answer and the updated graph state
        st.session_state.messages.append({"role": "Assistant", "content": response_state["current_response"]})
        st.session_state.chat_state = response_state

        index = get_feedback_interaction_index(len(st.session_state.messages)) - 1
        trace_manager.track_interaction(index, prompt, state["current_request"],
                                        response_state["current_response"])

    except Exception as e:
        st.error(f"Error: {str(e)}")
        message = "An error occurred. Please check the logs"
        st.session_state.messages.append({"role": "Assistant", "content": message})
        trace_manager.track_interaction(prompt, state["current_request"], message)
        logging.error(e, exc_info=True)

    finally:
        for p in temp_paths:
            if os.path.exists(p): os.remove(p)


# --- 5. UI LAYOUT ---
def main():
    setup_page()
    initialize_session()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # --- NEW: Conversation ID UI Element ---
        conv_id = trace_manager.conversation_id
        st.markdown(f"""
                    <div style="background-color: #1e2630; padding: 15px; border-radius: 10px; border: 1px solid #343d46; margin-bottom: 20px;">
                        <p style="color: #8892b0; font-size: 0.8rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Current Session ID</p>
                        <code style="color: #117aca; font-size: 1rem; font-weight: bold;">{conv_id}</code>
                    </div>
                """, unsafe_allow_html=True)

        model = st.selectbox("Choose model", ["gemma4:e4b", "phi3:mini"], disabled=st.session_state.processing)
        st.divider()
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True,
                                 disabled=st.session_state.processing)
        if st.button("Clear Session", disabled=st.session_state.processing):
            clear_session()

    # Main Chat
    st.subheader("PDF Assistant")
    status_container = st.empty()

    # Display History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 1. Check if the message is from the Assistant
            if msg["role"] == "Assistant":
                index = get_feedback_interaction_index(i)
                # 2. Only show buttons if feedback hasn't been given yet
                if index not in st.session_state.feedback_scores:
                    # Use columns to keep buttons small and side-by-side
                    col1, col2, _ = st.columns([0.05, 0.05, 0.9])

                    with col1:
                        if st.button("👍", key=f"up_{index}"):
                            # Log the positive feedback
                            st.session_state.feedback_scores[index] = "thumb_up"
                            st.rerun()  # Refresh to hide buttons

                    with col2:
                        if st.button("👎", key=f"down_{index}"):
                            # Log the negative feedback
                            st.session_state.feedback_scores[index] = "thumb_down"
                            st.rerun()  # Refresh to hide buttons

    # add the gif if processing is on going
    # Create an empty placeholder for the loading state
    loading_placeholder = st.empty()
    if st.session_state.processing:
        # Replace the placeholder with a GIF
        with loading_placeholder.container():
            st.markdown(
                """
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px;">
                    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueXF4ZzRnd3B6eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCBjdXN0b20mY3Q9Zw/3o7bu3XilJ5BOiSGic/giphy.gif" width="100">
                    <p style="color: #117aca; font-weight: bold; margin-top: 10px;">Bot is thinking...</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Execute the Graph
            current_prompt = st.session_state.pending_prompt
            try:
                bot = get_chatbot(model)  # 'model' comes from your sidebar selectbox
                temp_paths = process_uploaded_files(files)  # 'files' comes from your sidebar uploader

                run_pipeline(bot, current_prompt, temp_paths, status_container)
            except Exception as e:

                # st.stop()  # Safely stops execution without killing the app [6]
                logging.error(e)
                logging.shutdown()
                trace_manager.save_metadata()

            # 2. THE CLEANUP
            st.session_state.pending_prompt = None
            st.session_state.processing = False
            st.rerun()  # Final rerun to unlock the UI and show the new message

    else:
        # Clear the GIF immediately after response
        loading_placeholder.empty()

    # Input handling
    if prompt := st.chat_input("Ask a question...", disabled=st.session_state.processing):

        if len(prompt) > 10000:
            st.error(
                f"Input too long! Your request is {len(prompt)} characters, but the limit is 10,000. "
                f"Please shorten your content and question.")
            st.stop()  # Prevents the rest of the code from running

        else:
            st.session_state.messages.append({"role": "User", "content": prompt})
            # Store the pending prompt and lock the UI
            st.session_state.pending_prompt = prompt
            st.session_state.processing = True
            st.rerun()  # This stops execution and re-renders the UI with EVERYTHING disabled

    for each_key in st.session_state.feedback_scores.keys():
        trace_manager.track_user_feedback(each_key, st.session_state.feedback_scores[each_key])


if __name__ == "__main__":
    main()
    trace_manager.save_metadata()
