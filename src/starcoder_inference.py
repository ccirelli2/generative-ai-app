"""
Functions for generating code using Huggingface StarCoder.

References
=================================================
https://github.com/bigcode-project/starcoder
https://huggingface.co/bigcode/starcoder

"""
import ast
import re
import requests


def post_request(
        url: str,
        headers: dict,
        question: str,
        timeout: int = 60
):
    """
    Query the HuggingFace API.
    :param hf_url:
    :param headers:
    :param question: is of type str(dict)
    :param timeout:
    :return:
    """
    response = requests.post(
        url=url,
        headers=headers,
        json={"inputs": question},
        timeout=60
    )
    return response


def validate_response(response: str, question: str):
    response = response.json()[0]["generated_text"]
    response = response.replace(question, '').strip()
    response = re.sub(r'#.*', '', response)
    try:
        ast.parse(response)
        return response
    except Exception as err:
        print(err)
