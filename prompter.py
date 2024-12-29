import requests
import json
import yaml

# Constants for API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"  # Default URL for OpenRouter API
HF_URL = "https://api-inference.huggingface.co/v1/chat/completions"  # Alternate URL for HuggingFace API

class Prompter:
    """
    A class for making requests to LLM APIs
    """
    def __init__(
        self,
        base_url=OPENROUTER_URL,  # Default base URL for API requests
    ):
        """
        Initializes the Prompter instance with default or specified values.

        Args:
            base_url (str): The base URL of the API endpoint. Defaults to OpenRouter's URL.
        """
        self.base_url = base_url  # Base URL for API requests
        self.token = None  # API authorization token
        self.model_name = None  # Name of the model to use
        self.generation_args = {}  # Additional arguments for generation
        self.logged = False  # Reserved flag, possibly for logging activity (unused here)
        self.prompt_template = None

    # =============================================
    def _set_base_url(
        self,
        base_url: str,  # New base URL for API requests
    ):
        """
        Sets the base URL for API requests.

        Args:
            base_url (str): The new base URL for the API.
        """
        self.base_url = base_url
        return

    def _set_token(
        self,
        token: str,  # API authorization token
    ):
        """
        Sets the authorization token for API requests.

        Args:
            token (str): The API token.
        """
        self.token = token
        return

    def _set_model(
        self,
        model_name: str,  # Model identifier for the API
    ):
        """
        Sets the model name to use for requests.

        Args:
            model_name (str): The model identifier (e.g., "gpt-3.5-turbo").
        """
        self.model_name = model_name
        return

    def _set_generation_args(
        self,
        generation_args: dict = {},  # Additional generation parameters
    ):
        """
        Sets additional generation arguments to customize API behavior.

        Args:
            generation_args (dict): Dictionary of generation parameters (e.g., temperature, max_tokens).
        """
        self.generation_args = generation_args
        return

    def _update_generation_arg(
        self,
        key,
        value,
    ):
        """
        Update a key of the generation args
        """
        self.generation_args[key] = value
        return 
    # =============================================
    def generate(
        self,
        prompt_dicts: list[dict],  # List of message dictionaries defining the conversation
        stream: bool = False,  # Flag for streaming responses (not implemented in this method)
    ):
        """
        Sends a request to the API to generate a response based on the given prompts.

        Args:
            prompt_dicts (list[dict]): A list of message dictionaries containing the prompts.
            stream (bool): Whether to stream the response. Defaults to False.

        Returns:
            str: The content of the response message.
        """
        # Make a POST request to the API
        response = requests.post(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.token}",  # Authorization header with token
            },
            json={
                "model": self.model_name,  # Model name for the request
                "messages": prompt_dicts,  # Conversation history/messages
                **self.generation_args  # Additional generation arguments
            }
        )
        # Parse the API response and return the content of the first choice
        return json.loads(response.content)["choices"][0]["message"]["content"]

    def generate_batch(
        self,
        prompts: list,  # List of prompt strings
    ):
        """
        Generates responses for a batch of prompts by calling the `generate` method for each.

        Args:
            prompts (list): A list of prompt strings.

        Returns:
            list: A list of response strings for each prompt.
        """
        # TODO: Implement optimized batch requests (e.g., parallelization or API-specific batching)
        return [self.generate(prompt_dicts=[{"role": "user", "content": self.make_prompt(p)}]) for p in prompts]

    # =============================================
    def make_prompt(
        self, prompt:str,
    ):
        if not self.prompt_template is None:
            return self.prompt_template.format(text=prompt)
        return prompt

    def load_prompt_template(
        self,
        yaml_template:str=None,
    ):
        if yaml_template is None:
            self.prompt_template = None
        else:
            # Load the YAML file
            #with open(yaml_template, "r") as file:
            yaml_template = yaml.safe_load(yaml_template)

            prompt_tmp = "\n".join([
                yaml_template["prefix"], 
                yaml_template["core_prompt"], 
                yaml_template["suffix"]
            ])

            self.prompt_template = prompt_tmp

        return
