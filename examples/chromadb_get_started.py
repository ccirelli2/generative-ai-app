"""
For this tutorial we are not using the Langchain library in order to expose the class objects and functions native to
ChromaDB.

Notes
==================
Example of prompt engineering using a vector store.
For vector store we are going to use Chromadb.
This approach will require that the embeddings be created every time.

Chromadb
==================
- Step1: Get Chroma Client
    - chroma_client = chromadb.Client()
    - here you can pass in settings to the chroma client.
    - Settings (not all)
        - chroma_db_impl='chromadb.db.duckdb.DuckDB'
        - persist_directory='.chroma'

References
==================
- Chromadb: https://www.trychroma.com/
- Getting-Started: https://docs.trychroma.com/getting-started
- Usage: https://docs.trychroma.com/usage-guide
"""
# Import Libraries
import os
import logging
import chromadb
from decouple import config as d_config
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.vector_stores.embeddings import LangChainLoadChunkDocs

# Library Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_DATA = d_config("DIR_DATA")

# Globals
TEXT_FILE_NAME = "moby_dick.txt"


##################################################
# Tutorial
##################################################

# Step1: Define Chroma Client Settings (db & location to persist collectsion)
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="/home/christopher-cirelli/repositories/generative-ai-app/data/chroma_db",
)

# Step2: Create Chroma Client & Pass in Settings
chroma_client = chromadb.Client(settings)

# Step3: Get Collections
logger.info(f"List of Collections => {chroma_client.list_collections()}")

# Step4: Create Collection
collection = chroma_client.get_or_create_collection(
    name="moby-dick",
    embedding_function="sentence-transformers/all-MiniLM-L6-v2"
)

# Load & Chunk Text
# Load Documents: List[langchain.schema.Document]
loadChunkDoc = LangChainLoadChunkDocs(
    directory=DIR_DATA,
    file_name=TEXT_FILE_NAME
).run()

documents = loadChunkDoc.doc

"""
Does Chroma db require that we pass in embeddings or just the text?
"""

# Step 5: Create Embeddings
# sentences = ["This is an example sentence", "Each sentence is converted"]
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)



# Add Documents & Metadata
# collection.add(
#   embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
#   documents=["This is a document", "This is another document"],
#   metadatas=[{"source": "my_source"}, {"source": "my_source"}],
#   ids=["id1", "id2"]
# )

# Query DB
# collection.query(
#     query_texts=[1.2, 2.3, 4.5],
#     n_results=1
# )

#
# response = collection.query(query_embeddings=[1.2, 2.3, 4.5], n_results=1)
#
# print(response)