"""

"""
# Import Standard / Installed Libraries
import os
import streamlit as st
from decouple import config as d_config
from huggingface_hub import login

# Globals
DIR_ROOT = d_config("DIR_ROOT")
DIR_APP = os.path.join(DIR_ROOT, 'app')
os.chdir(DIR_ROOT)

# Import Custom Modules
from config import config_app, config_hugging_face, config_openai
from config import config_app as app_config

# Globals
HF_TOKEN = d_config("HUGGING_FACE_TOKEN")
OPENAI_TOKEN = d_config("OPEN_AI_TOKEN")

# Create Model Names Config
model_names_config = {"OpenAI": config_openai.model_names, "HuggingFace": config_hugging_face.model_names}


########################################################################################################################
# APPLICATION
########################################################################################################################

# SideBar - Model Selection
with st.sidebar.title("Module Options"):

    # Model Service Provider
    service_provider = st.sidebar.selectbox(label="Service-Provider", options=app_config.service_provider_names)

    # Model Parent Name
    model_root_name = st.sidebar.selectbox(label="Model-Parent-Name", options=model_names_config[service_provider])

    # Model-Type
    if service_provider == "OpenAI":
        model_sub_name = st.sidebar.selectbox(
            label="Model-Chile-Name",
            options=model_names_config[service_provider][model_root_name]["names"]
        )
    else:
        model_sub_name = "None"

    # Logging In


# Main Page - Title
st.title("Welcome to the Code Completion Module!")
st.divider()

# Main Page - Code Completion
st.subheader("Description")
with st.expander("Details"):
    st.markdown("""
    - Code completion is the use of an LLM to complete the code associated with a prompt.
    - For example, if we pass an LLM 'def hello_world('  we can ask it to complete this function.

    """)

# Model Information
st.subheader("Model Information")
with st.expander("Details"):
    st.markdown(f"""
    - Service Provider: {service_provider}
    - Model Parent Name: {model_root_name}
    - Model Child Name: {model_sub_name}
    - Model Cost: {model_names_config[service_provider][model_root_name]["cost"]}
    - Model Description: {model_names_config[service_provider][model_root_name]["description"]}

    """)




# Login to HuggingFace
"""
login(token=HF_TOKEN)
########################################################################################################################
# App
########################################################################################################################
st.title("Code Generator")
st.markdown("---")
st.subheader("Summary")
st.text('''
This page allows you to generate code using HuggingFace StarCoder.
The code however cannot be executed''')
st.text('''
Below you will be asked to input a question for which you would like to generate code.\n
You will have two options.
- The first is to choose from a list of prepared questions.
- The second is to utilize free form text to ask a question.''')
st.text('')
st.text('''
Once the question is selected the user will see the full prompt to be submitted to the LLM.
Thereafter, the code will be generated.''')
st.text('')
st.text('')

# User Input
st.subheader("User Input to Generate Code")
user_input_type = st.selectbox(
    "Select from a curated list or free form user input",
    ["Curated Selection", "Free Form Input"],
)
st.text("")
st.text("")

if user_input_type == "Curated Selection":
    st.subheader("Curated Selections")
    user_input = st.selectbox(
        label="Select from one of our prepared coding questions.",
        options=starcoder_config['QUESTIONS'],
        disabled=False
    )
    # Create Template
    if user_input:
        prompt_template = starcoder_config['PROMPT_TEMPLATE']
        prompt = prompt_template.format(question=user_input)
        st.text("")
        st.text("")
        st.markdown("**Code Prompt**")
        st.write(prompt)
        st.text("")
        st.text("")

elif user_input_type == "Free Form Input":
    # User Input
    user_input = st.text_input("Input a question for which you would like to generate code.", )

    # Create Template
    if user_input:
        prompt_template = starcoder_config['PROMPT_TEMPLATE']
        prompt = prompt_template.format(question=user_input)
        st.text("")
        st.text("")

# Generate Code
if prompt:
    st.subheader("Code Generatation")
    st.divider()
    st.write("Generating Code...")

    response = si.post_request(
        url=starcoder_config['URL'],
        headers=starcoder_config['HEADERS'],
        question=prompt,
        timeout=60
    )

    st.write(response.json()[0]["generated_text"])


"""