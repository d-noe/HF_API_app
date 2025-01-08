import streamlit as st

# ==============================================
# Sub-functions for modularity

def provider_selection(services_dict:dict, default_check:str="API:HF"):
    
    def on_change_checkbox(current_key):
        # Prevent all checkboxes from being unchecked
        if not any(st.session_state[key] for key in checkbox_keys if key != current_key):
            st.session_state[current_key] = True
            st.warning("At least one provider must be selected.", icon="⚠️")
            return
        # uncheck all checkboxes except current
        for key in checkbox_keys:
            if key != current_key:
                st.session_state[key] = False

    checkbox_keys = []

    for k in services_dict:
        key = k
        st.checkbox(f"{services_dict[k]['name']} API", key=key, value=key==default_check, on_change=on_change_checkbox, args=(key,))
        checkbox_keys.append(key)

    key = "API:Custom"
    st.checkbox('Other Endpoint', key=key, on_change=on_change_checkbox, args=(key,))
    checkbox_keys.append(key)

    if st.session_state.get("API:Custom", False):
        handle_custom_endpoint_ui()

    return checkbox_keys


def handle_custom_endpoint_ui():
    """
    Renders the UI for custom endpoints, with predefined options (password-protected) and manual entry.
    """
    if not st.session_state.get("authenticated", False):
        with st.expander("Admin Login"):
            password = st.text_input("Enter admin password:", type="password", key="admin_password")
            if st.button("Submit", key="auth_submit"):
                if password == st.secrets["admin"]["password"]:
                    st.session_state["authenticated"] = True
                    st.success("Authentication successful!")
                else:
                    st.error("Incorrect password. Please try again.")
        return

    # Retrieve custom endpoints from secrets
    custom_endpoints = st.secrets.get("custom_endpoints", {})
    if not custom_endpoints:
        st.warning("No predefined custom endpoints found in secrets.")
        return

    # Predefined custom endpoints dropdown
    predefined_option = st.selectbox("Choose a predefined custom endpoint:",
                                     options=["Enter manually"] + list(custom_endpoints.keys()),
                                     key="custom_predefined_dropdown")
    
    if predefined_option == "Enter manually":
        # Manual entry for custom base URL
        custom_base_url = st.text_input("Enter base URL", key="manual_custom_base_url")
        st.session_state["base_url"] = custom_base_url if custom_base_url else ""
    else:
        # Set predefined endpoint
        selected_base_url = custom_endpoints[predefined_option]
        st.session_state["base_url"] = selected_base_url
        st.session_state["AVAILABLE_MODELS"] = ["tgi"]

def handle_provider_configuration(services_dict:dict, checkbox_keys:list=[]):
    """
    Configure the base URL and available models based on the selected provider.
    """
    token_name = None

    if len(checkbox_keys):
        selected_checkbox = [k for k in checkbox_keys if st.session_state[k]][0]

        if selected_checkbox == "API:Custom":
            #custom_base_url = st.text_input("Enter base url", key="custom_base_url")
            #st.session_state["base_url"] = custom_base_url
            # Custom endpoint is handled via session_state updates in `handle_custom_endpoint_ui`
            #st.session_state["AVAILABLE_MODELS"] = [""]
            st.session_state["HELP_MESSAGE"] = ""
        else:
            token_name = selected_checkbox.split(":")[1]+"_API_TOKEN"
            st.session_state["base_url"] = services_dict[selected_checkbox]["base_url"]
            st.session_state["AVAILABLE_MODELS"] = services_dict[selected_checkbox]["available_models"]
            st.session_state["HELP_MESSAGE"] = services_dict[selected_checkbox]["help_message"]

    return token_name


def get_api_token(token_name):
    """
    Handle API token input and set up connection status.
    """
    api_key = None

    if token_name in st.secrets:
        api_key = st.secrets[token_name]
    else:
        api_key = st.text_input(
            'Enter API token:',
            help=st.session_state["HELP_MESSAGE"],
            label_visibility="visible",
            type='password',
        )

    return api_key

def log_api(prompter, api_key:str=None):
    if not api_key is None:
        try:
            if api_key:
                prompter._set_token(api_key)
                prompter._set_base_url(st.session_state["base_url"])
                st.session_state["log_status"] = True
            else:
                st.warning('Please enter your API token.', icon='⚠️')
                st.session_state["log_status"] = False
        except Exception as e:
            st.warning(f'Error connecting to API: {e}', icon='⚠️')
            st.session_state["log_status"] = False

        # Display API connection status
        log_status_str = "🟢 Connected" if st.session_state["log_status"] else "🔴 Disconnected"
        st.text(f"API Status: {log_status_str}")

    return


def set_generation_parameters(prompter, default_gen_args):
    """
    Set generation parameters (e.g., temperature, top-p, max_tokens).
    """
    st.subheader("Generation Parameters")
    for k, v in default_gen_args.items():
        selected_val = st.slider(v["name"], v["min"], v["max"], v["default"])
        prompter._update_generation_arg(k, selected_val)


# ==============================================
# Main function

def configure_api(prompter, services_dict: dict, generation_args: dict):
    """
    Sidebar configuration for model selection and generation parameters.
    """
    st.header("Configuration")

    # Step 1: Select the provider
    checkbox_keys = provider_selection(services_dict=services_dict)

    # Step 2: Handle provider-specific configuration
    token_name = handle_provider_configuration(services_dict=services_dict, checkbox_keys=checkbox_keys)

    # Step 3: Manage API token input
    api_key = get_api_token(token_name)
    log_api(prompter, api_key)

    # Step 4: Set generation parameters
    set_generation_parameters(prompter, generation_args)
