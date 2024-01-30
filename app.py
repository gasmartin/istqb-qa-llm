import argparse
import getpass
import os
import re
import uuid

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.huggingface_hub import HuggingFaceHubEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI


DOCUMENTS_PATH = "./docs"
BATCH_SIZE = 50


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key: ")


def load_documents():
    documents = []

    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.endswith(".pdf"):
            documents.extend(PyPDFLoader(os.path.join(DOCUMENTS_PATH, filename)).load())

    return documents


def split_documents_into_chunks(documents):
    def __clear_text(text):
        text = re.sub(r"[\n\r]", " ", text)
        text = re.sub(r'[\s]{2,}', " ", text)
        text = re.sub(r'[\.]{2,}', ".", text)
        return text

    document_chunks = CharacterTextSplitter(separator=".", chunk_size=2000, chunk_overlap=300).split_documents(documents)

    for chunk in document_chunks:
        chunk.page_content = __clear_text(chunk.page_content)

    return document_chunks


def get_vector_store(init_db):
    embeddings = HuggingFaceHubEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

    if init_db:
        print("Initializing vector database...")
        documents = load_documents()
        print("Spliting documents into document chunks...")
        document_chunks = split_documents_into_chunks(documents)
        print("Number of document chunks:", len(document_chunks))
        vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
        print("Creating embeddings for each document chunk and storing them...")
        print("Batch size:", BATCH_SIZE)
        for i in range(0, len(document_chunks), BATCH_SIZE):
            batched_documents = document_chunks[i : i + BATCH_SIZE]
            print(f"{i} ~ {i + len(batched_documents)}")
            texts = [doc.page_content for doc in batched_documents]
            metadatas = [doc.metadata for doc in batched_documents]
            ids = [str(uuid.uuid1()) for _ in texts]
            vector_store.add_texts(texts, metadatas, ids)
    else:
        vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

    return vector_store


@st.cache_resource
def load_qa_chain(init_db):
    print("Loading QA chain...")

    # Get the vector store instance
    vector_store = get_vector_store(init_db=init_db)

    # Use Gemini Pro as Conversational Model
    llm = GoogleGenerativeAI(model="gemini-pro")

    # Create a template for LLM prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know. Don't try to make up an answer. Use three sentences at maximum. Keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful answer:"""

    return RetrievalQA.from_chain_type(
        llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(template)}
    )


def handle_user_question(question):
    return st.session_state.qa_chain.invoke({"query": question})["result"]


def _get_args():
    parser = argparse.ArgumentParser(prog="ISTQB Q&A")
    parser.add_argument(
        "-i", "--init", 
        action="store_true",
        default=False,
        dest="init", 
        help="Initialize the vector database by parsing the PDF files and loading embeddings"
    )
    return parser.parse_args()


def main():
    st.set_page_config(page_title="ISTQB Q&A")

    args = _get_args()

    st.session_state.qa_chain = load_qa_chain(init_db=args.init)
    
    st.header("ISTQB Q&A")

    question = st.text_input("Insert a question about ISTQB content: ")

    if question:
        answer = handle_user_question(question)
        st.write(answer)


if __name__ == "__main__":
    main()
