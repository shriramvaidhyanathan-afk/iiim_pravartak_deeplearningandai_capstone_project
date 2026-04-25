import logging
import traceback
import hashlib

import streamlit as st
import os
import tempfile
from graph import ChatBot, ChatHistory
from redactor import TextRedactor
from traceabilitymanager import trace_manager, TraceabilityManager
from vector_store import VectorStore
from ai_abstractions import ModelConfig, Provider
from ai_factory import (
    build_provider_bundle,
    get_google_embedding_models,
    get_google_flash_chat_models,
)


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
def get_chatbot(provider_name, chat_model_name, embedding_model_name):
    """Initializes the ChatBot once and caches it across reruns."""
    config = ModelConfig(
        provider=Provider(provider_name),
        chat_model=chat_model_name,
        embedding_model=embedding_model_name,
    )
    # Isolate Chroma persistence by provider + embedding model to avoid
    # dimension collisions when switching between vendors/models.
    embedding_key = hashlib.md5(
        f"{provider_name}:{embedding_model_name}".encode("utf-8")
    ).hexdigest()[:12]
    persistent_path = f"./chroma_db/{provider_name}_{embedding_key}"

    provider_bundle = build_provider_bundle(config)
    vector_store = VectorStore(
        embeddings=provider_bundle.embeddings,
        persistent_path=persistent_path,
    )
    chat_bot = ChatBot(llm_client=provider_bundle.llm, vector_store=vector_store)
    trace_manager.model_name = chat_model_name
    return chat_bot.build_graph()


@st.cache_data(ttl=300)
def get_google_flash_models_for_ui():
    """Caches model discovery to avoid repeated API calls on every rerun."""
    return get_google_flash_chat_models()


@st.cache_data(ttl=300)
def get_google_embedding_models_for_ui():
    """Caches embedding model discovery to avoid repeated API calls on every rerun."""
    return get_google_embedding_models()


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
    if "status_banner" not in st.session_state:
        st.session_state.status_banner = None


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
    st.session_state.status_banner = None
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
        st.session_state.status_banner = None
        status_placeholder.empty()

    except Exception as e:
        logging.error(e, exc_info=True)
        message = f"There was a problem: {str(e)}"
        st.session_state.status_banner = message
        status_placeholder.error(message)


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

        provider_name = st.selectbox("Choose provider", ["ollama", "google"], disabled=st.session_state.processing)
        if provider_name == "ollama":
            chat_model = st.selectbox("Choose chat model", ["gemma4:e4b"], disabled=st.session_state.processing)
            embedding_model = st.selectbox("Choose embedding model", ["qwen3-embedding:4b"], disabled=st.session_state.processing)
        else:
            flash_models = []
            flash_fetch_error = None
            try:
                flash_models = get_google_flash_models_for_ui()
            except Exception as e:
                flash_fetch_error = str(e)

            if flash_fetch_error:
                st.warning(f"Could not auto-fetch Flash models: {flash_fetch_error}")
                chat_model = st.text_input(
                    "Google chat model id",
                    value="gemini-2.0-flash",
                    disabled=st.session_state.processing,
                    help="Use a model id available for your API key.",
                )
            else:
                if not flash_models:
                    st.warning("No Flash chat models were returned for this API key.")
                    chat_model = st.text_input(
                        "Google chat model id",
                        value="gemini-2.0-flash",
                        disabled=st.session_state.processing,
                    )
                else:
                    chat_model = st.selectbox(
                        "Google chat model id (Flash)",
                        flash_models,
                        disabled=st.session_state.processing,
                    )

            embedding_models = []
            embedding_fetch_error = None
            try:
                embedding_models = get_google_embedding_models_for_ui()
            except Exception as e:
                embedding_fetch_error = str(e)

            if embedding_fetch_error:
                st.warning(f"Could not auto-fetch embedding models: {embedding_fetch_error}")
                embedding_model = st.text_input(
                    "Google embedding model id",
                    value="text-embedding-004",
                    disabled=st.session_state.processing,
                    help="Use an embedding model id available for your API key.",
                )
            else:
                if not embedding_models:
                    st.warning("No embedding-capable models were returned for this API key.")
                    embedding_model = st.text_input(
                        "Google embedding model id",
                        value="text-embedding-004",
                        disabled=st.session_state.processing,
                    )
                else:
                    embedding_model = st.selectbox(
                        "Google embedding model id",
                        embedding_models,
                        disabled=st.session_state.processing,
                    )
        st.divider()
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True,
                                 disabled=st.session_state.processing)
        if st.button("Clear Session", disabled=st.session_state.processing):
            clear_session()

    # Main Chat
    st.subheader("PDF Assistant")
    status_container = st.empty()
    if st.session_state.status_banner:
        status_container.error(st.session_state.status_banner)

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
                bot = get_chatbot(provider_name, chat_model, embedding_model)
                temp_paths = process_uploaded_files(files)  # 'files' comes from your sidebar uploader

                run_pipeline(bot, current_prompt, temp_paths, status_container)
            except Exception as e:
                message = f"There was a problem: {str(e)}"
                st.session_state.status_banner = message
                status_container.error(message)
                logging.error(e, exc_info=True)
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