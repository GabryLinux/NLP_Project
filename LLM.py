
from abc import ABC, abstractmethod
from google import genai
from google.genai import types
import json
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from groq import Groq

from Formatter import Formatter, GemmaFormatter, LLamaFormatter

# Base class for LLMs
# A LLM is an entity that can generate text given a list of messages.
# Since the messages has to be formatted in a specific way for each LLM, 
# the LLM class also has a method to get the appropriate formatter for the LLM.
# Since LLMs can come from a family of models, the LLM class also has a method to set the 
# model to be used for generation.
class LLM(ABC):
    @staticmethod
    @abstractmethod
    def generate(messages) -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_formatter() -> Formatter:
        pass

    @staticmethod
    @abstractmethod
    def set_model(model_name: str):
        pass

# The Gemma implementation of the LLM class. 
# It uses the Google GenAI API to generate text.
# The default model is "gemma-3-27b-it", but it can be changed using the set_model method.
class GemmaLLM(LLM):
    _client = None
    _model = "gemma-3-27b-it"

    @staticmethod
    def _get_client():
        if GemmaLLM._client is None:
            with open("API_KEY.json", "r") as key_file:
                api_key = json.load(key_file)["GENAI_KEY"]
            GemmaLLM._client = genai.Client(
                api_key = api_key
            )
        return GemmaLLM._client
    
    @staticmethod
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=120),
        stop=stop_after_attempt(7),
        retry=retry_if_exception_type(Exception)
    )
    def generate(messages) -> str:
        return GemmaLLM._get_client().models.generate_content(
                model=GemmaLLM._model,
                contents=messages,
                config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=512,
                        top_p=1,
                    )
                ).text or ""
    
    @staticmethod
    def get_formatter() -> Formatter:
        return GemmaFormatter()
    
    @staticmethod
    def set_model(model_name: str):
        GemmaLLM._model = model_name
        


# The LLama implementation of the LLM class.
# It uses the Groq API to generate text.
# The default model is "llama-3.3-70b-versatile", 
# but it can be changed using the set_model method.
class LLamaLLM(LLM):
    _client = None
    _model = "llama-3.3-70b-versatile"

    @staticmethod
    def _get_client():
        if LLamaLLM._client is None:
            api_key = ""
            with open("API_KEY.json", "r") as key_file:
                api_key = json.load(key_file)["GROQ_KEY"]
            LLamaLLM._client = Groq(api_key=api_key)
        return LLamaLLM._client
    
    @staticmethod
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=120),
        stop=stop_after_attempt(7),
        retry=retry_if_exception_type(Exception)
    )
    def generate(messages) -> str:
        return LLamaLLM._get_client().chat.completions.create(
            model=LLamaLLM._model,
            messages=messages,
            temperature=0,
            max_completion_tokens=512,
            top_p=1,
            stream=False,
            stop=None
        ).choices[0].message.content or ""
    
    @staticmethod
    def get_formatter() -> Formatter:
        return LLamaFormatter()
    
    @staticmethod
    def set_model(model_name: str):
        LLamaLLM._model = model_name


# The LLM Evaluator is the LLM used to generate JSON analysis of the messages (for the critic) 
# and to evaluate the negotiation session.
# It is based on llama-3.3-70b, the largest and fastest (and free) model available for me.
class LLM_Evaluator(LLamaLLM):
    @staticmethod
    def generate(messages) -> str:
        actual__model = LLamaLLM._model
        LLamaLLM.set_model("llama-3.3-70b-versatile")
        message = LLamaLLM.generate(messages)
        LLamaLLM.set_model(actual__model)
        return message
