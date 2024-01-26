import getpass
import os

from langchain.chains import ConversationalRetrievalChain, QAGenerationChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI


DOCUMENTS_PATH = "./docs"


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key: ")


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
    document_chunks = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=0).split_documents(documents)

    print(len(document_chunks))

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vector store
    vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory="./chroma_db")
    vector_store.persist()

    # Create memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Use Gemini Pro as Conversational Model
    llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)

    qa_model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    question = input("Please insert your question: ")

    while question:
        answer = qa_model.invoke({"question": question})["answer"]
        print(answer)
        question = input("Please insert your question: ")


if __name__ == "__main__":
    main()
