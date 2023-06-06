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
from decouple import config as d_config

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader

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
"""
There are many different file types that can be loaded.  Langchain has document loaders
for many file types https://python.langchain.com/en/latest/modules/indexes/document_loaders.html.

Here, and for convinience, we have a function that determines the loader based on the file type.
"""
def load_document_by_type(directory: str, file_name: str, file_extension: str) -> list:
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


# Get Text Loader Object
loader = load_document_by_type(directory=DIR_DATA, file_name=PDF_FILE_NAME, file_extension=".pdf")


# Text Splitter
"""
Instantiate a text splitter object w/ parameters (there are many different types of text splitters).
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=20,
    length_function=len,
)

# Load & Split Text
"""
We load the text and split it into chunks at the same time.
This is convenient for when we create vector stores.
"""
documents = loader.load_and_split(text_splitter=text_splitter)
docN = documents[0]
