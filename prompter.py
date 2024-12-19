import abc
from huggingface_hub import InferenceClient

class Prompter:
    def __init__(
        self,
    ):
        self.logged = False

    @abc.abstractmethod
    def _get_token(
        self,
        **kwargs
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_client(
        self,
        **kwargs
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_generation_args(
        self,
        **kwargs
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        prompt_dicts:list[dict],
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def generate_batch(
        self,
        prompts:list,
    ):
        raise NotImplementedError

class PrompterHF(Prompter):
    def __init__(
        self,
    ):
        super().__init__()

    # =============================================
    def _set_client(
        self,
        token
    ):
        self.client = InferenceClient(token=token)

    def _set_model(
        self,
        model_name,
    ):
        self.model_name = model_name
        if hasattr(self, "client"):
            self.client.get_model_status(model_name)
        return

    def _set_generation_args(
        self,
        generation_args:dict={}
    ):
        self.generation_args = generation_args
        return

    # =============================================
    def generate(
        self,
        prompt_dicts:list[dict],
        stream:bool=False
    ):
        # Query the model
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt_dicts,
            #stream=stream,
            **self.generation_args
        )
        return response['choices'][0]['message']['content']

    def generate_batch(
        self,
        prompts:list,
    ):
        # TODO
        return [self.generate(prompt_dicts=[{"role":"user", "content":p}]) for p in prompts]
