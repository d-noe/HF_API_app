import os
import streamlit as st
import pandas as pd
from streamlit_float import *

from prompter import *
from utils import *
from app_config import *

float_init(theme=True, include_unstable_primary=False)

SERVICES = read_json("params/services.json")
DEFAULT_GEN_ARGS = read_json("params/gen_args.json")


def clear_conversation():
    st.session_state.messages = []

def set_log_status(status_bool:bool=True):
    st.session_state["log_status"] = status_bool


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
    uploaded_template = st.file_uploader("[Opt.] upload yaml prompt template:", type=["yaml"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if uploaded_template:
            prompter.load_prompt_template(uploaded_template.getvalue())
        st.write("Preview of uploaded file:", df.head())
        text_column = st.selectbox("Select Text Column for Completion:", df.columns)
        st.write(f"Prompt example:\n\n {prompter.make_prompt(list(df[text_column])[0])}")

        if st.button("Generate Completions"):
            if not st.session_state["log_status"]:
                st.error("⚠️ API is not connected. Please check your HuggingFace API token in the sidebar.")
            else:
                with st.spinner("Generating completions..."):
                    try:
                        df[f"{prompter.model_name}_completion"] = await prompter.generate_batch(prompts=df[text_column])
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
    st.title("LLMs APIs Interactions")  # Persistent title

    # Initialize prompter
    prompter = Prompter()

    # Configure sidebar
    with st.sidebar:
        configure_api(
            prompter = prompter,
            services_dict = SERVICES,
            generation_args = DEFAULT_GEN_ARGS
        )

    # Model selection
    available_models = st.session_state["AVAILABLE_MODELS"]
    available_models_w_custom = available_models + ["Custom Model (Enter Below)"]
    selected_option = st.selectbox("Select or Enter Model Name:", available_models_w_custom)

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
