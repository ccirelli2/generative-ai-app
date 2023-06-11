"""
page to create and persist embeddings in the chromadb and then to query then via qa functionality.

By query, we simply want to get the nearest neighbors of a given query string.
"""
# Import Standard / Installed Libraries
import os
import time
import openai
import chromadb
import streamlit as st
from decouple import config as d_config
from src.vector_stores.embeddings import LangChainLoadChunkDocs

# Globals
DIR_ROOT = d_config("DIR_ROOT")
DIR_DATA = d_config("DIR_DATA")
os.chdir(DIR_ROOT)

###################################################################
# Application
###################################################################
st.markdown("# ChromaDB - Create & Query Embeddings")
st.divider()

# Create Side Bar
with st.sidebar.markdown("## Collection Settings"):
    if st.sidebar.button("Connect to ChromaDB"):
        chroma_client = chromadb.Client()
        st.session_state['chroma_client'] = chromadb.Client()
        st.sidebar.success("Connected to ChromaDB")

        # Existing Collections
        st.sidebar.markdown("### Existing Collections")
        collection_election = st.sidebar.selectbox(label="Collections", options=chroma_client.list_collections())
        if not collection_election:
            st.sidebar.warning("No collections found.")

# Create Tabs
tab1, tab2 = st.tabs(["**Create Embeddings**", "**Query Embeddings**"])

if st.session_state.get('chroma_client'):
    # Tab1 - Create Embeddings
    with tab1:
        # Upload File
        st.markdown("### File Upload")
        st.caption("Create embeddings for a given text file.")
        uploaded_file = st.file_uploader(
            label="Upload Text File",
            type=["txt"],
            help="Only uploading text files is supported at this time."
        )

        if uploaded_file:
            st.success("File Uploaded Successfully")
            # Create Collection
            st.markdown("### Name Your Collection")
            collection_name = st.text_input("collection-name", help="""
            The length of the name must be between 3 and 63 characters.
            The name must start and end with a lowercase letter or a digit,
            The name CAN contain dots, dashes, and underscores in between.
            The name must NOT contain two consecutive dots.""", value="")
            if collection_name:
                collection_name = collection_name.lower()
                assert len(collection_name) > 3, "Collection name must be greater than 3 characters."
                assert '..' not in collection_name, "Collection name cannot contain two consecutive dots."
                assert collection_name not in (st.session_state['chroma_client'].list_collections(),
                                               "Collection name already exists.")
                st.caption(f"Collection name **{collection_name}** logged successfully.")

                # Button to Create Collection
                if st.button("Create Collection"):
                    st.caption("Creating collection...")

                    # Load & Chunk Text
                    # !!! Modify class object to take either the file or a path to the file.
                    """
                    documents = LangChainLoadChunkDocs(
                        directory=DIR_DATA,
                        file_name=TEXT_FILE_NAME,
                        chunk_size=250,
                        chunk_overlap=20,
                        length_function=len
                    ).run().doc

                    # Expose Documents, Metadata, & Create IDs.
                    sample_size = 10
                    document_text = [x.page_content for x in documents][:sample_size]
                    document_metadata = [x.metadata for x in documents][:sample_size]
                    document_ids = [f"chunk_{i}" for i in range(len(documents))][:sample_size]

                    # Add Text to Collection
                    collection.add(
                        # embeddings=[[1.2, 2.3, 4.5]],
                        documents=document_text,
                        metadatas=document_metadata,
                        ids=document_ids
                    )
                    """

    # Tab 2 - Query Embeddings
    with tab2:
        st.markdown("## Query Embeddings")
        st.caption("Query embeddings for a given text file.")
        st.caption("This will create a new collection in ChromaDB")


else:
    st.warning("Please connect to ChromaDB.")
