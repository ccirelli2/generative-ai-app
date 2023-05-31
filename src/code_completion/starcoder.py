"""
Functions for generating code using Huggingface StarCoder.

References
=================================================
https://github.com/bigcode-project/starcoder
https://huggingface.co/bigcode/starcoder
https://github.com/bigcode-project/starcoder/tree/main

"""
import ast
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StarcoderAPI:
    def __init__(self, url: str, headers: str, prompt: str, timeout=60):
        self.url = url
        self.headers = headers
        self.prompt = prompt
        self.timeout = timeout
        self.response_json = None
        self.response_clean = None
        logger.info("Starcoder API Initialized Successfully.")

    def pipeline(self):
        self.post_request()
        self.format_response()
        return self

    def post_request(self):
        """
        Post request to HuggingFace StarCoder API.

        returns a model response object.
        """
        response = requests.post(
            url=self.url,
            headers=self.headers,
            json={"inputs": self.prompt},
            timeout=self.timeout
        )
        assert response.status_code == 200, f"Response Failed with Status Code: {response.status_code}"
        logger.info("Response Successful.")
        self.response_json = response.json()

        return self

    def format_response(self):
        """

        Notes:
        - The response contains both the prompt and the generated text.
        - These values are separated by \n\n.  Therefore, we split on the \n\n and take everything after index 1.
        - Then we recombine the text so that it will print as a formatted string.
        - ast.parse will parse a string into an abstract syntax tree, which can be used to validate if the code is valid.
        :return:
        """
        # Get Generated Text
        response_text = self.response_json[0]["generated_text"]
        response_clean = "".join(response_text.split("\n\n")[0])

        try:
            ast.parse(response_clean)
            logger.info("Response Formatted Successfully.")
            self.response_clean = response_clean

        except Exception as err:
            logger.info(f"Response Failed to Format with Error: {err}")
            self.response_clean = response_clean

        return self
