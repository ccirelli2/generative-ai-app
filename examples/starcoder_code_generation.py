"""
https://github.com/bigcode-project/starcoder
https://huggingface.co/bigcode/starcoder

"""
import ast
import re
import json
import requests
from decouple import config as d_config
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = d_config("HUGGING_FACE_TOKEN")
HF_INF_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"




"""
# Login to HuggingFace Hub
print('Logging into HF Hub')
login(token=HF_TOKEN)
print('Successfully logged in')

"""Query the API"""
headers = {"Authorization": f"Bearer {HF_TOKEN}"}
question = {"inputs": "Generate a function to add two values.  The parameters to the functions are a and b.  Return the sum of a and b."}

response = requests.post(
    url=HF_INF_URL,
    headers=headers,
    json=question,
    timeout=60
)

response = response.json()[0]["generated_text"]
response = response.replace(question['inputs'], '').strip()
response = re.sub(r'#.*', '', response)
try:
    ast.parse(response)
    print(response)
except Exception as err:
    print(err)

"""
