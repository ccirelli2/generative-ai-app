"""
Purpose is to test out pandasai on a test dataset.
To test out prompt engineering to be able to query tables and create charts.

References
=======================
1. https://dev.to/ngonidzashe/chat-with-your-csv-visualize-your-data-with-langchain-and-streamlit-ej7
2. https://python.langchain.com/en/latest/modules/agents/toolkits/examples/pandas.html

"""


# Import Libraries
import os
import logging
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from config import data_file_names
from langchain import OpenAI
from pandasai import PandasAI
from pandasai.llm.starcoder import Starcoder
from langchain.agents import create_pandas_dataframe_agent
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
HF_TOKEN = d_config("HUGGING_FACE_TOKEN")
HOUSING_FILE_NAME = data_file_names["HOUSING_FILE_NAME"]

# Load Data
housing_df = pd.read_csv(os.path.join(DIR_DATA, "housing.csv"))
shape = housing_df.shape
logger.info(f"Housing Dataframe dimensions => {shape}")

# Pandas AI
"""
llm = Starcoder(api_token=HF_TOKEN)
pandas_ai = PandasAI(llm)
pandas_ai._verbose = True
logger.info("Successfully instantiated pandas ai with HuggingFace Starcoder")
"""

# Langchain Pandas Agent




if __name__ == "__main__":
    pd_agent = create_pandas_agent(
        file_name=HOUSING_FILE_NAME,
        dir_data=DIR_DATA,
        llm_name="OPENAI",
        llm_token=OPENAI_TOKEN
    )

    """
    query = "how many rows are in this dataframe?"
    response = create_query_agent(pd_agent, query)
    print(f"Response: {response}")
    """