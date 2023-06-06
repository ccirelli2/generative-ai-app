"""

Notes
================
- LLM Prompt Template
    - Can be passed both the in-context parameter key value pairs as well as memory key value pairs if available.
    - - Allows you to define examples to be passed to the LLM (method from_examples).

References
================
- https://python.langchain.com/en/latest/modules/chains/generic/llm_chain.html
"""
import os
from langchain import PromptTemplate, OpenAI, LLMChain




print(help(PromptTemplate))