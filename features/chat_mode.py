import streamlit as st
from prompter import Prompter

def chat_mode(
    prompters:list[Prompter],
    max_histories:int=5,
    chat_input_key:str="chat_input",
    conv_histories_key:str="messages_histories",
    clear_button_key:str="clear_history"
):
    """Chat mode tab functionality with streaming."""
    num_models = len(prompters)
    st.subheader("Chat with the Model" if num_models==1 else "Chat with the Models")

    if conv_histories_key not in st.session_state:
        st.session_state[conv_histories_key] = [[] for _ in range(max_histories)]

    # Display chat messages
    if num_models == 1:
        for message in st.session_state[conv_histories_key][0]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        user_messages = [msg["content"] for msg in st.session_state[conv_histories_key][0] if msg["role"] == "user"]
        for turn_index, user_message in enumerate(user_messages):
            st.chat_message("user").write(user_message)
            columns = st.columns(num_models)
            for model_index, col in enumerate(columns):
                with col:
                    display_name = ""
                    try: 
                        display_name = f"{prompters[model_index].model_name.split('/')[1]}"
                    except:
                        display_name = f"Model {model_index+1}"
                    st.write(f"**{display_name}**")
                    assistant_messages = [msg["content"] for msg in st.session_state[conv_histories_key][model_index] if msg["role"] == "assistant"]
                    if turn_index < len(assistant_messages):
                        st.chat_message("assistant").write(assistant_messages[turn_index])

   # User input
    def add_message():
        user_message = st.session_state[chat_input_key].strip()
        if user_message:
            for i, history in enumerate(st.session_state[conv_histories_key][:num_models]):
                if not st.session_state["log_status"]:
                    history.append({
                        "role": "assistant",
                        "content": "⚠️ API is not connected. Please check your API token.",
                    })
                    return
                history.append({"role": "user", "content": user_message})
                try:
                    with st.spinner(f"Generating response for Model {i + 1}..." if num_models > 1 else "Generating response..."):
                        model_response = prompters[i].generate(prompt_dicts=history)
                        history.append({"role": "assistant", "content": model_response})
                except Exception as e:
                    history.append({"role": "assistant", "content": f"⚠️ Error while generating response: {e}"})

    st.chat_input("Type your message", key=chat_input_key, on_submit=add_message)
    st.button("Clear Chat", key=clear_button_key, on_click=lambda: st.session_state.update({conv_histories_key: [[] for _ in range(max_histories)]}))