"""

"""

import logging
import pandas as pd
import openai
from pprint import pprint
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


def openai_model_config(api_key: str) -> dict:
    """

    :param api_key:
    :return:
    """
    # Pass Open AI Personal Token
    openai.api_key = OPENAI_TOKEN
    # Get a list of all model data
    models = openai.Model.list()
    model_names = [x['id'] for x in models['data']]
    # Build Config
    model_groups = ["gpt-3", "ada", "babbage", "curie", "davinci"]
    model_config = {}
    for mg in model_groups:
        model_config[mg] = [x for x in model_names if mg in x]
    return model_config


if __name__ == "__main__":
    model_config = openai_model_config(api_key=OPENAI_TOKEN)
    pprint(model_config)