"""
Example code to create embedding.

Notes
==================
- Langchain TextLoader:
    - Class object to load text files.
    - https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
- langchain.schema.Document
    - An Interface for interacting with a document.
- Langchain Vector Stores:
    - By defaultLangChain uses Chroma as the vectorstore to index and search embeddings.


References


==================
- Langchain Indexes: https://python.langchain.com/en/latest/modules/indexes/getting_started.html

"""
# Import Libraries
import os
from pprint import pprint
from decouple import config as d_config

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_DATA = d_config("DIR_DATA")

# Globals
TEXT_FILE_NAME = "moby_dick.txt"
os.environ["OPENAI_API_KEY"] = d_config("OPEN_AI_TOKEN")

# Load Text
loader = TextLoader(os.path.join(DIR_DATA, TEXT_FILE_NAME), encoding='utf8')
documents = loader.load()  # returns a list of documents.
doc_n = documents[0]  # returns a langchain.schema.Document
doc_schema = doc_n.metadata  # document metadata.
doc_text = doc_n.page_content  # how to get raw text back from document.

# Create an Index
index = VectorstoreIndexCreator().from_loaders([loader])

query = "Who is Ahab?"
response = index.query(query)

print(response)

"""
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
"""