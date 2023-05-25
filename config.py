"""

"""
from langchain import PromptTemplate
from decouple import config as d_config


# Globals
HF_TOKEN = d_config("HUGGING_FACE_TOKEN")
HOUSING_DATA_FILE_NAME = {"HOUSING_FILE_NAME": "housing.csv"}

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
STARCODER_CONFIG = {
    "HEADERS": {"Authorization": f"Bearer {HF_TOKEN}"},
    "URL": "https://api-inference.huggingface.co/models/bigcode/starcoder",
    "QUESTIONS": starcoder_questions,
    "PROMPT_TEMPLATE": prompt_template,
}

question = "Write a function using the pandas library to create a pandas dataframe.  Make sure the function returns the dataframe."
prompt = prompt_template.format(question=question)
print(prompt)