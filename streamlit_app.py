import os
import streamlit as st
import pandas as pd
from streamlit_float import *

from prompter import *

float_init(theme=True, include_unstable_primary=False)

HELP_HF_TOKEN_API = """**Don't have an API token?** Head over to [HuggingFace](https://huggingface.co/docs/hub/security-tokens) to sign up for one."""

AVAILABLE_OR_MODELS = [
    "meta-llama/llama-3.2-1b-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "meta-llama/llama-3.1-70b-instruct:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
]

AVAILABLE_HF_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/QwQ-32B-Preview",
    "Qwen/Qwen2.5-72B-Instruct",
    "google/gemma-1.1-7b-it",
    "google/gemma-2-27b-it",
    "meta-llama/Llama-3.2-3B-Instruct",
    "NousResearch/Hermes-3-Llama-3.1-8B",
    "HuggingFaceH4/zephyr-7b-beta",
    "HuggingFaceH4/zephyr-7b-alpha",
    "01-ai/Yi-1.5-34B-Chat",
]

def clear_conversation():
    st.session_state.messages = []

def set_log_status(status_bool:bool=True):
    st.session_state["log_status"] = status_bool

def configure_sidebar(prompter):
    """Sidebar configuration for model selection and generation parameters."""
    st.sidebar.header("Configuration")

    # API Token
    if 'HF_API_TOKEN' in st.secrets:
        hf_api = st.secrets['HF_API_TOKEN']
    else:
        hf_api = st.sidebar.text_input('Enter HuggingFace API token:', help=HELP_HF_TOKEN_API, label_visibility="visible", type='password')

    try:
        if hf_api and (hf_api.startswith("hf_") and len(hf_api)==37):
            prompter._set_client(hf_api)
            st.session_state["log_status"] = True
        else:
            st.sidebar.warning('Please enter your HuggingFace API token.', icon='⚠️')
            st.session_state["log_status"] = False
    except Exception as e:
        st.sidebar.warning(f'Error connecting to HuggingFace API: {e}', icon='⚠️')
        st.session_state["log_status"] = False

    # Display API connection status
    log_status_str = "🟢 Connected" if st.session_state["log_status"] else "🔴 Disconnected"
    st.sidebar.text(f"API Status: {log_status_str}")

    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    temperature = st.sidebar.slider("Temperature:", 0.0, 2.0, 1.0)
    top_p = st.sidebar.slider("Top-p:", 0.0001, .9999, .9)
    max_tokens = st.sidebar.slider("Max New Tokens:", 1, 2048, 512)
    prompter._set_generation_args({"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens})


def chat_mode(prompter):
    """Chat mode tab functionality with streaming."""
    st.subheader("Chat with HuggingFace Model")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

   # User input
    def add_message():
        user_message = st.session_state.chat_input.strip()
        if user_message:
            st.session_state.messages.append({"role": "user", "content": user_message})
            if not st.session_state["log_status"]:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "⚠️ API is not connected. Please check your HuggingFace API token in the sidebar.",
                })
                return
            with st.spinner("Generating response..."):
                try:
                    model_response = prompter.generate(prompt_dicts=st.session_state.messages)
                    st.session_state.messages.append({"role": "assistant", "content": model_response})
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Error while generating response: {e}",
                    })

    st.chat_input("Type your message", key="chat_input", on_submit=add_message)
    st.button("Clear Chat", on_click=lambda: st.session_state.update({"messages": []}))


def csv_upload_mode(prompter):
    """CSV upload and completion mode tab functionality."""
    st.subheader("Upload CSV and Get Completions")
    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded file:", df.head())
        text_column = st.selectbox("Select Text Column for Completion:", df.columns)

        if st.button("Generate Completions"):
            if not st.session_state["log_status"]:
                st.error("⚠️ API is not connected. Please check your HuggingFace API token in the sidebar.")
            else:
                with st.spinner("Generating completions..."):
                    try:
                        df[f"{prompter.model_name}_completion"] = prompter.generate_batch(prompts=df[text_column])
                        st.success("Completions added!")
                        st.write("Preview of updated file:", df.head())

                        # Provide download link for updated CSV
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download Updated CSV",
                            data=csv,
                            file_name="completed_file.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"⚠️ Error while generating completions: {e}")


def main():
    # Ensure the title is displayed on every re-render
    st.title("HuggingFace Model Interaction")  # Persistent title

    # Initialize prompter
    prompter = PrompterHF()

    # Configure sidebar
    configure_sidebar(prompter)

    # Model selection
    AVAILABLE_MODELS_WITH_CUSTOM = AVAILABLE_MODELS + ["Custom Model (Enter Below)"]
    selected_option = st.selectbox("Select or Enter Model Name:", AVAILABLE_MODELS_WITH_CUSTOM)

    if selected_option == "Custom Model (Enter Below)":
        custom_model_name = st.text_input("Enter Custom Model Name:", key="custom_model_input")
        selected_model = custom_model_name
    else:
        selected_model = selected_option

    if selected_model:
        try:
            prompter._set_model(selected_model)
            st.success(f"Model set to: {selected_model}")
        except Exception as e:
            st.error(f"Failed to set model: {e}")

    # Tabs for mode selection
    tab1, tab2 = st.tabs(["Chat with Model", "Upload CSV for Completion"])
    with tab1:
        chat_mode(prompter)
    with tab2:
        csv_upload_mode(prompter)


if __name__ == "__main__":
    main()
