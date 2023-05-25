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
from src.code_generation import starcoder as si
from config.starcoder import starcoder_config

# Globals
HF_TOKEN = d_config("HUGGING_FACE_TOKEN")
HF_API = starcoder_config['API']
prompt = None

# Login to HuggingFace
login(token=HF_TOKEN)

########################################################################################################################
# App
########################################################################################################################
st.title("Code Generator")
st.markdown("---")
st.subheader("Summary")
st.text("""
This page allows you to generate code using HuggingFace StarCoder.
The code however cannot be executed""")
st.text("""
Below you will be asked to input a question for which you would like to generate code.\n
You will have two options.
- The first is to choose from a list of prepared questions.
- The second is to utilize free form text to ask a question.""")
st.text("")
st.text("""
Once the question is selected the user will see the full prompt to be submitted to the LLM.
Thereafter, the code will be generated.""")
st.text("")
st.text("")

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

st.sidebar