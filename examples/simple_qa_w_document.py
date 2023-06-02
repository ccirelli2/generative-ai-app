"""
Example code to create embedding.

Notes
==================
- Langchain TextLoader:
    - Class object to load text files.
    - https://python.langchain.com/en/latest/modules/indexes/document_loaders.html
- langchain.schema.Document
    - An Interface for interacting with a document.
    - page_content method will return a string of the document.
    - the document is loaded by page, which is why we need to use the text_splitter for the vector store.
- Langchain Vector Stores:
    - By defaultLangChain uses Chroma as the vectorstore to index and search embeddings.
- PyPDFLoader
    - Loads a PDF with pypdf and chunks at character level.
- Text Splitters
    - Approach: Split text into sentences, then aggregate sentences until you reach a character limit.  Once you reach
      the character limit, aggregate that into a chunk.
    - The default recommended text splitter is the RecursiveCharacterTextSplitter
    - By default the characters it tries to split on are ["\n\n", "\n", " ", ""]
    - Parameters: length_function, chunk_size, chunk_overlap
    - Ref: https://python.langchain.com/en/latest/modules/indexes/text_splitters.html
    - Ref: https://python.langchain.com/en/latest/modules/indexes/text_splitters/getting_started.html

References
==================
- Langchain Indexes: https://python.langchain.com/en/latest/modules/indexes/getting_started.html

"""
# Import Libraries
import os
import logging
from pprint import pprint
from decouple import config as d_config

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Directories
DIR_ROOT = d_config("DIR_ROOT")
DIR_DATA = d_config("DIR_DATA")

# Globals
TEXT_FILE_NAME = "moby_dick.txt"
PDF_FILE_NAME = "aig_app.pdf"
os.environ["OPENAI_API_KEY"] = d_config("OPEN_AI_TOKEN")


# Load Text
def load_document(directory: str, file_name: str, file_extension: str) -> list:
    """

    doc_n = documents[0]  # returns a langchain.schema.Document
    doc_schema = doc_n.metadata  # document metadata.
    doc_text = doc_n.page_content  # how to get raw text back from document.

    :param directory:
    :param file_name:
    :param file_extension:
    :return:
    """
    logger.info(f"Loading file with extension => {file_extension}")
    if file_extension == ".txt":
        loader = TextLoader(os.path.join(directory, file_name), encoding='utf8')
    elif file_extension == ".pdf":
        loader = PyPDFLoader(os.path.join(directory, file_name))
    else:
        raise Exception("File extension not recognized")
    logger.info("Loading finished")
    return loader

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
)

loader = load_document(directory=DIR_DATA, file_name=PDF_FILE_NAME, file_extension=".pdf")

# text_loaded = loader.load()  # returns a list of langchain.schema.Document indexed by page number.
# docN = text_loaded[1]
# print(dir(docN))
# print(docN.metadata)
# print(docN.page_content)

text_loaded = loader.load_and_split()
docN = text_loaded[0]
print(docN.page_content)

"""

print(dir(loader))
print(loader.source)
print(loader.file_path)



# Create an Index
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What is a limit of Liability?"
response = index.query(query)

print(response)
"""
