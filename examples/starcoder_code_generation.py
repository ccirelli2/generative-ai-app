"""
https://github.com/bigcode-project/starcoder
https://huggingface.co/bigcode/starcoder

"""
from huggingface_hub import login
from src.code_completion.starcoder import StarcoderAPI
from config.config_hugging_face import api, models
from config import config_prompts

# Login to HuggingFace Hub
login(token=api["token"])

# Prompt
prompt = "def create_pandas_dataframe(data: dict):"

# API
starApi = StarcoderAPI(
    url=models["starcoder"]["api"],
    headers=api["headers"],
    prompt=prompt,
    timeout=60
).pipeline()

print(starApi.response_clean)

"""
response_json = response.json()
response_text = response_json[0]["generated_text"]
response_clean = "".join(response_text.split("\n\n")[1:])
print(response_clean)

response = response.json()[0]["generated_text"]
response = response.replace(question['inputs'], '').strip()
response = re.sub(r'#.*', '', response)
try:
    ast.parse(response)
    print(response)
except Exception as err:
    print(err)
"""
