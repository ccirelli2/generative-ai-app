"""

Notes
==========================
- Get List of all models: openai.Model.list()

References
==========================
- Open-AI-Docs: https://platform.openai.com/docs/api-reference/authentication?lang=python

"""
# Import Libraries
import logging
import pandas as pd
import openai
from decouple import config as d_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_DATA = d_config("DIR_DATA")

# Globals
OPENAI_TOKEN = d_config("OPEN_AI_TOKEN")
OPENAI_MODEL = 'text-davinci-003'

# Log Into Open AI
openai.api_key = OPENAI_TOKEN

# Instantiate LLM Model
#llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2, temperature=0.9)

# Text completion
def prompt_completion(
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        echo: bool = False
) -> str:
    """
    function to generate text or code completion.

    :param model:
    :param prompt:
    :param max_tokens:
    :param temperature: range from 0 to 2. 0 is the most conservative and 2 is the most random.
    :param echo: whether to echo the prompt in the response
    :return:
    """
    assert max_tokens <= 50, "To control costs max tokens cannot exceed 20."

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=echo
    )

    return response

def concat_prompt_response(prompt: str, response: dict) -> str:
    """
    Function to combine the prompt and the text response.
    :param response:
    :return:
    """
    return prompt + response['choices'][0]['text']


if __name__ == "__main__":
    max_tokens = 50
    temperature = 0.5
    prompt = "def hello_world(name:"

    response = prompt_completion(
        model=OPENAI_MODEL,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )

    complete_response = concat_prompt_response(prompt=prompt, response=response)
    print(complete_response)
