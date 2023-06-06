# Import Libraries
import os
import logging

from decouple import config as d_config

from langchain.indexes import VectorstoreIndexCreator

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Import Project Modules
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
os.environ["OPENAI_API_KEY"] = d_config("OPEN_AI_TOKEN")

if __name__ == "__main__":

    # Load Embeddings Model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    """
    # Load Documents: List[langchain.schema.Document]
    loadChunkDoc = LoadChunkDoc(
        directory=DIR_DATA,
        file_name=TEXT_FILE_NAME
    ).run()

    documents = loadChunkDoc.doc

    # Instantiate Index Object
    fil = {"persist_directory": "/home/christopher-cirelli/repositories/generative-ai-app/data/chroma_db",}
    index = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        embedding=hf_embeddings,
        vectorstore_kwargs=fil,
    ).from_documents(documents)

    #print(help(VectorstoreIndexCreator.from_loaders))
    """
    vectorstore = Chroma(
        collection_name="debug",
        persist_directory="/home/christopher-cirelli/repositories/generative-ai-app/data/chroma_db",
        collection_metadata={"source-file": TEXT_FILE_NAME},
        embedding_function=hf_embeddings
    )

    print(help(vectorstore))


