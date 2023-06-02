"""
Notes
==================
Example of prompt engineering using a vector store.
For vector store we are going to use Chromadb.
This approach will require that the embeddings be created every time.

Chromadb
==================
- It appears that Chromadb will create embeddings for you if you provide a text field.

References
==================
1. Chromadb: https://www.trychroma.com/
"""
import os
import chromadb

# Instantiate Chroma DB
chroma_client = chromadb.Client()

# Create a  Collection
collection = chroma_client.create_collection("my_collection")

# Add Documents & Metadata
collection.add(
    documents=["This is a document", "This is another document"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)

# Query DB
results = collection.query(
    query_texts=["This is a query document"],
    n_results=1
)

print(results)

