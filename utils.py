import os
import json
import yaml

def read_json(path:str):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def get_available_templates(templates_folder:str):
    # Initialize template options list
    template_options = []
    template_descriptions = {}

    # Fetch YAML files in the templates folder and extract name and description
    if os.path.exists(templates_folder):
        for filename in os.listdir(templates_folder):
            if filename.endswith(".yaml"):
                with open(os.path.join(templates_folder, filename), 'r') as file:
                    template = yaml.safe_load(file)
                    template_options.append(filename)
                    template_descriptions[filename] = template.get('description', 'No description available')

    return template_options, template_descriptions
