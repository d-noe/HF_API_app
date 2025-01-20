import streamlit as st
import pandas as pd
from prompter import Prompter
from app_config import *
from streamlit_hf_plugged import SERVICES, DEFAULT_GEN_ARGS

from features.chat_mode import *
from features.csv_mode import *


# def multimodels_chatmode(prompters:list[Prompter]):
#     # Chat mode interface
#     st.subheader("Chat Mode: Multi-Model Comparison")
#     num_models = len(prompters)

#     def handle_user_input():
#         user_input = st.session_state.user_multi_input.strip()
#         if user_input:
#             for i, history in enumerate(st.session_state.multi_model_messages[:num_models]):
#                 history.append({"role": "user", "content": user_input})
#                 try:
#                     with st.spinner(f"Generating response for Model {i + 1}..."):
#                         model_response = prompters[i].generate(history)
#                         history.append({"role": "assistant", "content": model_response})
#                 except Exception as e:
#                     history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})

#     # Display the conversation for all models in a single unified chat format
#     st.write("### Conversation View")
#     user_messages = [msg["content"] for msg in st.session_state['multi_model_messages'][0] if msg["role"] == "user"]

#     for turn_index, user_message in enumerate(user_messages):
#         st.chat_message("user").write(user_message)
#         columns = st.columns(num_models)
#         for model_index, col in enumerate(columns):
#             with col:
#                 display_name = ""
#                 try: 
#                     display_name = f"{model_configs[model_index]['model_name'].split('/')[1]}"
#                 except:
#                     display_name = f"Model {model_index+1}"
#                 st.write(f"**{display_name}**")
#                 assistant_messages = [msg["content"] for msg in st.session_state['multi_model_messages'][model_index] if msg["role"] == "assistant"]
#                 if turn_index < len(assistant_messages):
#                     st.chat_message("assistant").write(assistant_messages[turn_index])

#     # Unique user input
#     user_input = st.chat_input("Type your message", key="user_multi_input", on_submit=handle_user_input)
#     # Clear conversation button
#     if st.button("Clear Conversation"):
#         st.session_state.multi_model_messages = [[] for _ in range(5)]

# =============================================================================

def multimodels_compare():
    """
    Multi-model comparison feature with chat interface and CSV completion mode.
    """
    # Initialize session states for conversation and model configuration
    if 'model_comparison' not in st.session_state:
        st.session_state['model_comparison'] = {
            'num_models': 2,
            'models': []
        }

    # Title and configuration
    st.header("LLM Output Comparison Tool")
    st.subheader("Model Comparison Configuration")

    # Model selection and configuration
    num_models = st.slider("Select number of models to compare:", 2, 5, value=st.session_state['model_comparison']['num_models'])
    st.session_state['model_comparison']['num_models'] = num_models

    # Create UI for selecting models and setting parameters
    # =====================================================
    # st.write("### Model Configuration")
    # model_configs = []

    # columns = st.columns(num_models)
    # for i, col in enumerate(columns):
    #     with col:
    #         st.write(f"#### Model {i + 1}")
    #         base_url = st.text_input(f"Model {i + 1} Base URL:", key=f"base_url_{i}")
    #         model_name = st.text_input(f"Model {i + 1} Name:", key=f"model_name_{i}")
    #         api_token = st.text_input(f"API Token for Model {i + 1}:", key=f"api_token_{i}", type="password")
    #         temperature = st.slider(f"Temperature for Model {i + 1}:", 0.0, 1.0, 0.7, key=f"temperature_{i}")
    #         max_tokens = st.number_input(f"Max Tokens for Model {i + 1}:", min_value=1, max_value=4096, value=256, key=f"max_tokens_{i}")

    #         model_configs.append({
    #             'base_url': base_url,
    #             'model_name': model_name,
    #             'api_token': api_token,
    #             'temperature': temperature,
    #             'max_tokens': max_tokens
    #         })
    # Instantiate Prompter instances for each model
    # prompters = [Prompter(config['base_url']) for config in model_configs]
    # for p, c in zip(prompters, model_configs):
    #     p._set_token(c['api_token'])
    #     p._set_model(c['model_name'])
    #     p._set_generation_args({
    #         "temperature": c['temperature'],
    #         "max_new_tokens": c['max_tokens']
    #     })
    # =====================================================

    columns = st.columns(num_models)
    # Initialize prompters
    prompters = [Prompter() for _ in range(num_models)]
    for i, col in enumerate(columns):
        # Configure sidebar
        with col:
            configure_api(
                prompter = prompters[i],
                services_dict = SERVICES,
                generation_args = DEFAULT_GEN_ARGS,
                key_suffix=f"_multi_{i}",
            )

            # Model selection
            available_models = st.session_state["AVAILABLE_MODELS"]
            available_models_w_custom = available_models + ["Custom Model (Enter Below)"]
            selected_option = st.selectbox("Select or Enter Model Name:", available_models_w_custom, key=f"model_selection_multi_{i}")

            if selected_option == "Custom Model (Enter Below)":
                custom_model_name = st.text_input("Enter Custom Model Name:", key=f"custom_model_name_{i}")
                selected_model = custom_model_name
            else:
                selected_model = selected_option

            if selected_model:
                try:
                    prompters[i]._set_model(selected_model)
                    st.success(f"Model set to: {selected_model}")
                except Exception as e:
                    st.error(f"Failed to set model: {e}")

    # Mode selection: Chat or CSV
    st.write("### Select Mode")
    mode = st.radio("Choose interaction mode:", ["Chat Mode", "CSV Completion Mode"], index=0)

    if mode == "Chat Mode":
        #multimodels_chatmode(prompters)
        chat_mode(
            prompters=prompters,
            max_histories=5,
            chat_input_key="chat_input_multi",
            conv_histories_key="messages_histories_multi",
            clear_button_key="clear_multi_histories"
        )

    elif mode == "CSV Completion Mode":
        csv_upload_mode(
            prompters=prompters,
            csv_upload_key="csv_upload_multi",
            yaml_upload_key="yaml_upload_multi",
            select_template_key="template_selection_multi",
            select_column_key="column_to_complete_multi",
            download_completed_key="download_completed_multi"
        )



# import streamlit as st
# import pandas as pd
# from prompter import Prompter


# def multimodels_compare():
#     # Initialize session state to store comparison settings and conversation histories
#     if 'model_comparison' not in st.session_state:
#         st.session_state['model_comparison'] = {
#             'num_models': 2,
#             'models': [],
#             'responses': []
#         }

#     if 'conversation_histories' not in st.session_state:
#         st.session_state['conversation_histories'] = [[] for _ in range(5)]  # Support up to 5 models

#     # Clear conversation button
#     if st.sidebar.button("Clear Conversation"):
#         st.session_state['conversation_histories'] = [[] for _ in range(5)]

#     # App title
#     st.title("LLM Output Comparison Tool")

#     # Sidebar for selecting number of models
#     st.sidebar.header("Model Comparison Configuration")
#     num_models = st.sidebar.slider("Select number of models to compare:", 2, 5, value=st.session_state['model_comparison']['num_models'])
#     st.session_state['model_comparison']['num_models'] = num_models

#     # Create UI for selecting models and setting parameters
#     st.write("### Model Configuration")
#     model_configs = []

#     for i in range(num_models):
#         st.write(f"#### Model {i + 1}")
#         base_url = st.text_input(f"Model {i + 1} Base URL:", key=f"base_url_{i}")
#         model_name = st.text_input(f"Model {i + 1} Name:", key=f"model_name_{i}")
#         api_token = st.text_input(f"API Token for Model {i + 1}:", key=f"api_token_{i}", type="password")
#         temperature = st.slider(f"Temperature for Model {i + 1}:", 0.0, 1.0, 0.7, key=f"temperature_{i}")
#         max_tokens = st.number_input(f"Max Tokens for Model {i + 1}:", min_value=1, max_value=4096, value=256, key=f"max_tokens_{i}")

#         model_configs.append({
#             'base_url': base_url,
#             'model_name': model_name,
#             'api_token': api_token,
#             'temperature': temperature,
#             'max_tokens': max_tokens
#         })

#     # Mode selection: Chat or CSV
#     st.write("### Select Mode")
#     mode = st.radio("Choose interaction mode:", ["Chat Mode", "CSV Completion Mode"], index=0)

#     # Instantiate Prompter instances for each model
#     prompters = [Prompter(config['base_url']) for config in model_configs]
#     for p, c in zip(prompters, model_configs):
#         p._set_token(c['api_token'])
#         p._set_model(c['model_name'])
#         p._set_generation_args(
#             {
#                 "temperature": c['temperature'],
#                 "max_new_tokens": c['max_tokens']
#             }
#         )

#     if mode == "Chat Mode":
#         st.write("### Chat Interface")
#         user_input = st.text_input("Enter your prompt:", key="user_input_chat", label_visibility='collapsed')

#         if user_input and st.button("Send Prompt"):
#             responses = []
#             for i, prompter in enumerate(prompters):
#                 # Append user input to conversation history
#                 st.session_state['conversation_histories'][i].append({"role": "user", "content": user_input})

#                 # Generate response
#                 response = prompter.generate(st.session_state['conversation_histories'][i])
#                 st.session_state['conversation_histories'][i].append({"role": "assistant", "content": response})

#         # Display the conversation for all models in a single unified chat format
#         st.write("### Conversation View")
#         user_messages = [msg["content"] for msg in st.session_state['conversation_histories'][0] if msg["role"] == "user"]

#         for turn_index, user_message in enumerate(user_messages):
#             st.chat_message("user").write(user_message)

#             columns = st.columns(num_models)
#             for model_index, col in enumerate(columns):
#                 with col:
#                     st.write(f"### Model {model_index + 1}")
#                     assistant_messages = [msg["content"] for msg in st.session_state['conversation_histories'][model_index] if msg["role"] == "assistant"]
#                     if turn_index < len(assistant_messages):
#                         st.chat_message("assistant").write(assistant_messages[turn_index])

#     elif mode == "CSV Completion Mode":
#         st.write("### CSV Completion")
#         uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"])

#         if uploaded_file:
#             input_df = pd.read_csv(uploaded_file)
#             st.write("### Uploaded Data")
#             st.dataframe(input_df)

#             if st.button("Send Completion Requests"):
#                 for i, prompter in enumerate(prompters):
#                     st.write(f"#### Processing with Model {i + 1}")
#                     input_df[f"Model_{i + 1}_Output"] = input_df['prompt'].apply(lambda prompt: prompter.generate([{"role": "user", "content": prompt}]))

#                 st.write("### Results")
#                 st.dataframe(input_df)

#                 # Option to download the results
#                 csv = input_df.to_csv(index=False).encode('utf-8')
#                 st.download_button("Download Results as CSV", data=csv, file_name="model_comparison_results.csv", mime='text/csv')
