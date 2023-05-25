"""
functions related to the langchain pandas agent.

References
=======================
- https://python.langchain.com/en/latest/modules/agents/toolkits/examples/pandas.html
"""
import os
import logging
from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_pandas_agent(
        file_name: str,
        dir_data: str,
        llm_name: str,
        llm_token: str,
        verbose: bool = True
):
    """
    Create an agent that can access and use a large language model (LLM).

    :param verbose:
    :param llm_token:
    :type llm_name: object
    :param file_name:
    :param dir_data:

    ::Returns:
        An agent that can access and use a large language model (LLM).
    """
    logger.info("Create Pandas Agent")

    # Instantiate LLM
    if llm_name.upper() == "OPENAI":
        llm = OpenAI(openai_api_key=llm_token, temperature=0.0)
        logger.info(f"Instantiating {llm_name.upper()} LLM")
        logger.info(f"Type => {type(llm)}")
    elif llm_name.upper() == "STARCODER":
        llm = Starcoder(api_token=llm_token)
        logger.info(f"Instantiating {llm_name.upper()} LLM")
        logger.info(f"Type => {type(llm)}")
    else:
        raise ValueError("Invalid LLM name")

    # Read DataFrame
    logger.info("Reading DataFrame")
    df = pd.read_csv(os.path.join(dir_data, file_name))

    # Create Pandas DataFrame Agent
    df_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=verbose
    )
    logger.info("Successfully created Pandas DataFrame Agent")
    return df_agent


def create_query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    # Create the prompt.
    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: {query}
            """
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()
