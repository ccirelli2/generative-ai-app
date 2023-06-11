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

# Define Chroma Client Settings (db & location to persist collection)
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="/home/christopher-cirelli/repositories/generative-ai-app/data/chroma_db",
)

# Create Chroma Client & Pass in Settings
chroma_client = chromadb.Client(settings)
chroma_client.reset()

# Lost Existing Collections
logger.info(f"List of  Collections => {chroma_client.list_collections()}")

# Step4: Create Collection
collection_name = "test-1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
    # embedding_function="sentence-transformers/all-MiniLM-L6-v2"
)
logger.info(f"Loading collection => {collection_name} with number of records => {collection.count()}")

collection.add(
    #embeddings=[[1.2, 2.3, 4.5]],
    documents=['today is a nice day to code'],
    metadatas=[{'source': 'my-mind'}],
    ids=["id3"]
)

# Query DB
response = collection.query(
    query_texts=["tomorrow is a good day to code"],
    n_results=1
)
#
#
print(response)

