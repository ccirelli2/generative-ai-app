"""
source code for creating embeddings.
"""
# Import Libraries
import os
import logging
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.document_loaders import TextLoader

# Library Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LangChainLoadChunkDocs:
    def __init__(
            self,
            directory: str,
            file_name: str,
            chunk_size: int = 256,
            chunk_overlap: int = 20,
            length_function=len
    ):
        """

        :param directory:
        :param file_name:
        """
        self.directory = directory
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.loader = None
        self.loader_config = {"txt": TextLoader, "pdf": PyPDFLoader}
        self.text_splitter = None
        self.doc = None
        logger.info("LoadChunkDocs Instantiated Successfully")

    def run(self):
        self._loader()
        self._splitter()
        self._load_chunk()
        logger.info(f"Pipeline LoadChunkDocs Completed Successfully.  Returning {len(self.doc)} chunks.\n\n")
        return self

    def _loader(self):
        """
        Dynamically loads the correct document loader based on the file extension and loads the doc.
        :return:
        """
        logger.info(f"Instantiating Loader")
        file_extension = self.file_name.split(".")[-1]
        assert file_extension in self.loader_config.keys(), f"File extension {file_extension} not recognized."
        self.loader = self.loader_config[file_extension](os.path.join(self.directory, self.file_name))
        return self

    def _splitter(self):
        """

        :return:
        """
        logger.info(f"Instantiating Text Splitter with Parameters")
        logger.info(f"\t\tChunk Size => {self.chunk_size}")
        logger.info(f"\t\tChunk Overlap => {self.chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
        )
        return self

    def _load_chunk(self):
        logger.info(f"Loading & Splitting Document")
        self.doc = self.loader.load_and_split(text_splitter=self.text_splitter)
        return self
