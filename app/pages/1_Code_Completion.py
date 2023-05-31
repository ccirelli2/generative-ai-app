"""

"""
# Import Standard / Installed Libraries
import os
import openai
import streamlit as st
from decouple import config as d_config
from huggingface_hub import login

# Globals
DIR_ROOT = d_config("DIR_ROOT")
os.chdir(DIR_ROOT)

# Import Custom Modules
from config import config_hugging_face, config_openai, config_prompts
from config import config_app as app_config
from src.code_completion.starcoder import StarcoderAPI

# Globals
HF_TOKEN = d_config("HUGGING_FACE_TOKEN")
OPENAI_TOKEN = d_config("OPEN_AI_TOKEN")

# Create Model Names Config
model_config = {"HuggingFace": config_hugging_face.models, "OpenAI": config_openai.models}
api_config = {"HuggingFace": config_hugging_face.api, "OpenAI": config_openai.api}


########################################################################################################################
# APPLICATION
########################################################################################################################

#######################################################
# SideBar - Model Selection
#######################################################
with st.sidebar.title("Module Options"):
    # Caption
    st.sidebar.markdown("Please select the foundational model provider and model below. :face_with_monocle:")
    st.sidebar.caption("")

    # Model Service Provider
    service_provider = st.sidebar.selectbox(label="Service-Provider", options=app_config.service_provider_names)

    # Model Parent Name
    model_parent_name = st.sidebar.selectbox(label="Model-Parent-Name", options=model_config[service_provider])

    # Model-Type
    if service_provider == "OpenAI":
        model_sub_name = st.sidebar.selectbox(
            label="Model-Child-Name",
            options=model_config[service_provider][model_parent_name]["names"]
        )
    else:
        model_sub_name = "None"

    # Logging In
    st.sidebar.caption("**Log into service provider**")
    if st.sidebar.button(label=f"Login to {service_provider}"):

        st.sidebar.caption(f"Logging into {service_provider}...")

        if service_provider == "HuggingFace":
            try:
                login(token=HF_TOKEN)
                st.sidebar.caption("Login successful :partying_face:")
            except Exception as e:
                st.sidebar.caption(f"Login failed :face_with_head_bandage:\n{e}")

        elif service_provider == "OpenAI":
            try:
                openai.api_key = OPENAI_TOKEN
                st.sidebar.caption("Login successful :partying_face:")
            except Exception as e:
                st.sidebar.caption(f"Login failed :face_with_head_bandage:\n{e}")


#######################################################
# Main Page - Model Information
#######################################################
st.title("Welcome to the Code Completion Module!")
st.divider()

# Main Page - Code Completion
st.subheader("Module Description")
with st.expander("Details"):
    st.markdown("""
    - Code completion is the use of an LLM to complete the code associated with a prompt.
    - For example, if we pass an LLM 'def hello_world('  we can ask it to complete this function.
    """)

# Model Information
st.subheader("Model Information")
with st.expander("Details"):
    st.markdown(f"""
    - **Service Provider**: {service_provider}
    - **Model Parent Name**: {model_parent_name}
    - **Model Child Name**: {model_sub_name}
    - **Model Cost**: {model_config[service_provider][model_parent_name]["cost"]}
    - **Model Description**: {model_config[service_provider][model_parent_name]["description"]}  
    """)

#######################################################
# Generator
#######################################################
st.subheader("Code Generator Description")
with st.expander("Details"):
    st.markdown('''
    The generator allows you to complete your code using a prompt.
    A prompt is an incomplete piece of text or code for which you would like the LLM to complete.
    
    **Example**:
    
    - Prompt: def hello_world():
    - Response: def hello_world(): print('hello world')
    
    **Options**:
    - The first is to choose from a list of prepared questions.
    - The second is to utilize free form text to ask a question.
    
    Once the prompt is submitted to the LLM you will receive a response with the models best guess on the complete code.
    ''')
    st.text('')
    st.text('')

#######################################################
# Prompt Selection
#######################################################
st.subheader("Prompt Selection")
prompt = st.selectbox(label="Prompts", options=config_prompts.code_completion)

if st.button(label="Complete Code"):
    starApi = StarcoderAPI(
        url=model_config[service_provider]["api"],
        headers=api_config[service_provider]["headers"],
        prompt=prompt,
        timeout=60
    ).pipeline()

    st.write(starApi.response_clean)