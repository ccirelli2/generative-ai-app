"""
Configurations for Starcoder LLM.

TODO I think we should make the template configurable from the application.  here we are hard coding a single option.

"""
from langchain import PromptTemplate
from decouple import config as d_config


# StarCoder Questions
starcoder_questions = [
    "",
    "Write a function to print hello world.",
    "Write a function to return 'hello world'",
]

# StarCoder Prompt Template
prompt = '''{question}'''
prompt_template = PromptTemplate(input_variables=["question"], template=prompt)


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
