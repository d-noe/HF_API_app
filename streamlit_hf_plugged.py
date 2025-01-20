import os
import streamlit as st
import pandas as pd
from streamlit_float import *

from prompter import *
from utils import *
from app_config import *
from features.chat_mode import *
from features.csv_mode import *
from features.multimodels import *

float_init(theme=True, include_unstable_primary=False)

SERVICES = read_json("params/services.json")
DEFAULT_GEN_ARGS = read_json("params/gen_args.json")


def clear_conversation():
    st.session_state.messages = []

def set_log_status(status_bool:bool=True):
    st.session_state["log_status"] = status_bool


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
    tab1, tab2, tab3 = st.tabs(["Chat with Model", "Upload CSV for Completion", "🚧 Compare Models"])
    with tab1:
        #chat_mode(prompter)
        chat_mode(
            prompters=[prompter],
            max_histories=1,
            chat_input_key="chat_input_unique",
            conv_histories_key="messages_history",
            clear_button_key="clear_history"
        )
    with tab2:
        #csv_upload_mode(prompter)
        csv_upload_mode(
            prompters=[prompter],
            csv_upload_key="csv_upload",
            yaml_upload_key="yaml_upload",
            select_template_key="template_selection",
            select_column_key="column_to_complete",
            download_completed_key="download_completed"
        )
    with tab3:
        multimodels_compare()


if __name__ == "__main__":
    main()
