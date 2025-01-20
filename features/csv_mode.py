import streamlit as st
from prompter import Prompter
import pandas as pd
from utils import *

def csv_upload_mode(
    prompters:list[Prompter],
    csv_upload_key:str="csv_upload",
    yaml_upload_key:str="yaml_upload",
    select_template_key:str="template_selection",
    select_column_key:str="column_to_complete",
    download_completed_key:str="download_completed"
):
    """CSV upload and completion mode tab functionality."""
    num_models = len(prompters)

    def report_error_to_streamlit(row_index, error_message):
        st.error(f"Row {row_index}: {error_message}")

    st.subheader("Upload CSV and Get Completions")
    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"], key=csv_upload_key)
    uploaded_template = st.file_uploader("[Opt.] upload yaml prompt template:", type=["yaml"], key=yaml_upload_key)
    # Dropdown to select pre-existing templates from the templates folder
    templates_folder = "./templates/"
    template_options, template_descriptions = get_available_templates(templates_folder)
    selected_template_file = st.selectbox(
        "Or choose an existing template from the list below:",
        options=[None] + template_options,
        format_func=lambda x: x if x is None else x,
        key=select_template_key
    )

    if selected_template_file:
        # Display template name and description
        with open(os.path.join(templates_folder, selected_template_file), 'r') as file:
            template = yaml.safe_load(file)
            st.write(f"**Template Name:** {template.get('name', 'Unnamed Template')}")
            st.write(f"**Description:** {template_descriptions[selected_template_file]}")
        # Load the chosen template
        for p in prompters:
            p.load_prompt_template(open(os.path.join(templates_folder, selected_template_file)).read())

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if uploaded_template:
            for p in prompters:
                p.load_prompt_template(uploaded_template.getvalue())
        st.write("Preview of uploaded file:", df.head())
        text_column = st.selectbox("Select Text Column for Completion:", df.columns, key=select_column_key)
        # Display prompt example
        st.write(f"**Prompt example:**\n\n {prompters[0].make_prompt(list(df[text_column])[0])}")

        # Default completion column
        default_completion_cols = [f"{p.model_name}_completion" for p in prompters]
        # Allow user to select or specify an existing completion column
        st.write("**Select or specify the column for existing completions:**")
        columns = st.columns(num_models)
        completion_columns = ["" for _ in range(num_models)]
        for i, col in enumerate(columns):
            with col:
                completion_columns [i] = st.text_input(
                    "Completion Column:", 
                    value=default_completion_cols[i], 
                    help="Leave as default or specify a column name where completions are stored."
                )
        # Initialize the completion column if it doesn't exist
        for cc in completion_columns:
            if cc not in df.columns:
                st.warning(f"Column '{cc}' does not exist. It will be created.")
                df[cc] = ""
            else:
                st.warning(f"Column '{cc}' alredy exists. It will be overwritten.")

        # Identify rows with missing completions
        missing_indices = [df[df[cc].isna() | (df[cc] == "")].index for cc in completion_columns]
        if num_models == 1:
            st.write(f"Rows with missing completions: {len(missing_indices[0])}")
        else:
            missing_strs = [f"- {cc} : {len(mi)}" for mi, cc in zip(missing_indices, completion_columns)]
            st.write("Rows with missing completions:\n\n\t"+"\n\n\t".join(missing_strs))

        if st.button("Generate Completions"):
            if not st.session_state["log_status"]:
                st.error("⚠️ API is not connected. Please check your HuggingFace API token in the sidebar.")
            else:
                for i, prompter in enumerate(prompters):
                    with st.spinner("Generating completions..." if num_models==1 else f"Generating completions for Model {i+1}..."):
                        try:
                            # Generate completions only for rows with missing values
                            prompts = df.loc[missing_indices[i], text_column].tolist()
                            completions = prompter.generate_batch(prompts=prompts, **{'error_callback':report_error_to_streamlit})
                            # Update the DataFrame with new completions
                            for idx, completion in zip(missing_indices[i], completions):
                                df.at[idx, completion_columns[i]] = completion

                            st.success("Completions added!" if num_models==1 else f"Completions added for Model {i+1}!")
                        except Exception as e:
                            st.error(f"⚠️ Error while generating completions: {e}")

                # Provide download link for updated CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Updated CSV",
                    data=csv,
                    file_name="completed_file.csv",
                    mime="text/csv",
                    key=download_completed_key
                )
                # Preview csv
                st.write("Preview of updated file:", df.head())


def multimodels_csvcompletion(prompters):
    st.write("### CSV Completion")
    uploaded_file = st.file_uploader("Upload CSV File:", type=["csv"], key="file_multi_gen")

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.dataframe(input_df)

        if st.button("Send Completion Requests"):
            for i, prompter in enumerate(prompters):
                st.write(f"#### Processing with Model {i + 1}")
                input_df[f"Model_{i + 1}_Output"] = input_df['prompt'].apply(lambda prompt: prompter.generate([{"role": "user", "content": prompt}]))

            st.write("### Results")
            st.dataframe(input_df)

            # Option to download the results
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name="model_comparison_results.csv", mime='text/csv')
