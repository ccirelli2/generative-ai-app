"""
Open ai functions for code generation.
"""
# Import Libraries
import logging
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def format_response(prompt: str, response: dict) -> str:
    """
    Function to combine the prompt and the text response.
    :param prompt:
    :param response:
    :return:
    """
    return prompt + response['choices'][0]['text']
