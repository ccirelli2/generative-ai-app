"""
Example code to create embedding.


Questions
==================
- Does this approach include creating embeddings?  I don't see where we create embeddings prior to the index and usage
of Chroma.

Notes
==================
- Indexes:
    _ Langchain focused on constructing indexes with the goal of using them as a Retriever.
    - Focus on VectorStore retriever.
    - By default langchain uses Chroma as the vector store to index and search embeddings.
- Langchain TextLoader:
    - Class object to load text files.
    - https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
- langchain.schema.Document
    - An Interface for interacting with a document.
- Langchain Vector Stores:
    - By defaultLangChain uses Chroma as the vectorstore to index and search embeddings.
- VectorStoreIndexCreator
    - Class object to create an index.
    - Ability to pass in a document loader object, text splitter and text embedding objects.
    - Instead of using the default settings we are going to manually select the text splitter and embeddings
    models.  This will be helpful for future cases where we want to deviate from the default settings and to better
    understand the usage of this class object.
    - Two methods
        - from_documents
            - documents: List[langchain.schema.Document]
            - this means that we can create the documents in advance using our own text splitter and chunk size and then
            pass that to this model.
        - from_loaders
            - loaders: List[lanchain.document_loaders.base.BaseLoader]
            - meaning that we can pass in the load object before creating the documents.
- Chroma DB
    - When creating a collection without passing an embeddings model the following log message is generated
        instantiating ChromaNo embedding_function provided,
        using default embedding function:
        DefaultEmbeddingFunction https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
    - This is a nice feature as we do not need to specify an embeddings model, but we do need to know the parameters
    that we want to pass, i.e. the size of the embeddings to create.
    - Error Message line 387: https://github.com/hwchase17/chat-langchain/issues/57
- Sentence Transformers
    - all-MiniLM-L6-v2
        - This is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space
        and can be used for tasks like clustering or semantic search.

References
==================
- Langchain Indexes: https://python.langchain.com/en/latest/modules/indexes/getting_started.html
- Langchain Embeddings: https://python.langchain.com/en/latest/reference/modules/embeddings.html#
- HF Sentence Transformers: https://huggingface.co/sentence-transformers
- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Langchain Vector Store Notebooks: https://github.com/hwchase17/langchain/blob/master/docs/modules/indexes/getting_started.ipynb
"""
# Import Libraries
import os
import logging
from pprint import pprint
from decouple import config as d_config

from langchain import llms
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer

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


# Step1: Load Text & Chunk
"""
Here we want to load our text document using the Langchain text-loader object and chunk the text all in one shot.
- We have a class object that will take in the file and automatically read and chunk it into a langchain document.
"""


# Step 2: Create Embeddings
"""
# Create an Index
index = VectorstoreIndexCreator().from_loaders([loader])

query = "Who is Ahab?"
response = index.query(query)

print(response)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
"""


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

    # Load Documents: List[langchain.schema.Document]
    loadChunkDoc = LangChainLoadChunkDocs(
        directory=DIR_DATA,
        file_name=TEXT_FILE_NAME
    ).run()

    documents = loadChunkDoc.doc

    # Instantiate Index Object
    index = VectorstoreIndexCreator(
        embedding=hf_embeddings
    ).from_documents(documents)

    #print(help(VectorstoreIndexCreator.from_loaders))
