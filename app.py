import os

import torch
import transformers
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModel


DOCUMENTS_PATH = "./docs"


def load_documents():
    documents = []

    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.endswith(".pdf"):
            documents.extend(PyPDFLoader(os.path.join(DOCUMENTS_PATH, filename)).load())

    return documents


def main():
    # Load documents from folder
    documents = load_documents()

    # Split documents into chunks
    document_chunks = CharacterTextSplitter(chunk_size=50, chunk_overlap=0).split_documents(documents)

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="neuralmind/bert-base-portuguese-cased")

    # Create vector store
    vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory="./chroma_db")
    vector_store.persist()

    # Create memory
    # memory = ConversationBufferMemory(memory_key="bstqb_chat_memory", return_messages=True)


if __name__ == "__main__":
    main()
