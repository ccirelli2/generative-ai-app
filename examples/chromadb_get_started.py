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

# Create Chroma Client & Pass in Settings
chroma_client = chromadb.Client()
chroma_client.reset()

# Lost Existing Collections
logger.info(f"List of  Collections => {chroma_client.list_collections()}")

# Create Collection
collection_name = "moby-dick"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)
logger.info(f"Loading collection => {collection_name} with number of records => {collection.count()}")

# Load & Chunk Text
documents = LangChainLoadChunkDocs(
    directory=DIR_DATA,
    file_name=TEXT_FILE_NAME,
    chunk_size=250,
    chunk_overlap=20,
    length_function=len
).run().doc

# Expose Documents, Metadata, & Create IDs.
sample_size= 10
document_text = [x.page_content for x in documents][:sample_size]
document_metadata = [x.metadata for x in documents][:sample_size]
document_ids = [f"chunk_{i}" for i in range(len(documents))][:sample_size]

# Add Text to Collection
collection.add(
    #embeddings=[[1.2, 2.3, 4.5]],
    documents=document_text,
    metadatas=document_metadata,
    ids=document_ids
)



# # Query DB

# response = collection.query(
#     query_texts=["Who is Ahab?"],
#     n_results=1
# )
#
# print(response)

