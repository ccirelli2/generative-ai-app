"""
Configurations for Starcoder LLM.

TODO I think we should make the template configurable from the application.  here we are hard coding a single option.

"""
from langchain import PromptTemplate
from decouple import config as d_config

# Configs
models = {
    "starcoder": {
        "names": ["starcoder"],
        "cost": "Free",
        "description": "StarCoder was trained on GitHub code, thus it can be used to perform code generation."
                       "More precisely, the model can complete the implementation of a function or infer"
                       "the following characters in a line of code."
                       "The StarCoder models are 15.5B parameter models trained on 80+ programming languages from The"
                       "Stack (v1.2), with opt-out requests excluded. The model uses Multi Query Attention, a context"
                       "window of 8192 tokens, and was trained using the Fill-in-the-Middle objective on 1"
                       "trillion tokens."
                       "Intended use The model was trained on GitHub code. As such it is not an instruction model and"
                       "commands like 'Write a function that computes the square root.' do not work well. However, by"
                       "using the Tech Assistant prompt you can turn it into a capable technical assistant.",
        "url": "https://huggingface.co/bigcode/starcoder",
        "api": "https://api-inference.huggingface.co/models/bigcode/starcoder",
    },
    "starcoder_tech_assistant": {
        "names": ["starcoder_tech_assistant"],
        "cost": "Free",
        "description": "",
        "url": "https://huggingface.co/datasets/bigcode/ta-prompt",
        "api": "https://api-inference.huggingface.co/models/bigcode/ta-prompt",
    }
}

api = {
    "token": d_config("HUGGING_FACE_TOKEN"),
    "headers": {"Authorization": f"Bearer {d_config('HUGGING_FACE_TOKEN')}"}
}



"""
# StarCoder Prompt Template
#prompt = '''{question}'''
#prompt_template = PromptTemplate(input_variables=["question"], template=prompt)


# StarCoder Config
starcoder_config = {
    "TOKEN": d_config("HUGGING_FACE_TOKEN"),
    "HEADERS": {"Authorization": f"Bearer {d_config('HUGGING_FACE_TOKEN')}"},
    "API": "https://api-inference.huggingface.co/models/bigcode/starcoder",
    "URL": "https://huggingface.co/bigcode/starcoder",
    "QUESTIONS": starcoder_questions,
    "PROMPT_TEMPLATE": prompt_template,
    "DESCRIPTION": "StarCoder is a language model trained on a large corpus of Python code. It can be used to"
                   "generate code given a natural language description of the code to be generated.",
    "USE": "Code generation.  It is not an instruction model"
}
"""
