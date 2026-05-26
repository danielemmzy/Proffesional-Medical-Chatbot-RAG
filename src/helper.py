from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path):
    loader = DirectoryLoader(pdf_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
                Document(
            page_content=doc.page_content,
            metadata={"source": src}
                )
        )
    return minimal_docs

# split documents into chunks
def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunks = text_splitter.split_documents(minimal_docs)
    return texts_chunks


def download_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"
    return HuggingFaceEmbeddings(model_name=model_name)

embeddings = download_embeddings()