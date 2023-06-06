"""

Notes
==================
- HuggingFaceHub
    - Wrapper around HuggingFaceHub  models.
    - To use, you should have the huggingface_hub python package installed, and the
    environment variable HUGGINGFACEHUB_API_TOKEN
    set with your API token, or pass it as a named parameter to the constructor.
    - ***Only supports `text-generation`, `text2text-generation` and `summarization` for now.

- LLM Chain
    - Used to query an LLM Model.
    - It formats the prompt template using the input key value pairs, and also memory key values if available.
    - apply method allows you to run the chain against a list of questions or inputs.

# References
==================
- Langchain library for connecting to HF Hub
    https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html
- LLM Chain
    https://python.langchain.com/en/latest/modules/chains/generic/llm_chain.html
    https://python.langchain.com/en/latest/modules/chains/getting_started.html

"""
# Libraries
import os
from pprint import pprint
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from decouple import config as d_config
from langchain.output_parsers import CommaSeparatedListOutputParser

# Globals
HF_TOKEN = d_config("HUGGING_FACE_TOKEN")

# Select a Model
repo_id = "google/flan-t5-xl"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0, "max_length": 64},
    huggingfacehub_api_token=HF_TOKEN
)

# Create Prompt Template
template = """Question: {question}

Answer: Let's think step by step."""

# Create Prompt
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create LLM Chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Define Question
question = "Who won the FIFA World Cup in the year 1994? "

# Generate Prediction
response = llm_chain.predict(question=question)
print(response, '\n', '\n')

# Use Output Parser
output_parser = CommaSeparatedListOutputParser()
prompt = PromptTemplate(
    template=template,
    input_variables=["question"],
    output_parser=output_parser
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
response = llm_chain.predict_and_parse(question=question)
print(f"Response w/ Parser => {response}")
