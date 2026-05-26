from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from src.helper import extract_text_from_pdf, filter_to_minimal_docs, text_splitter, download_embeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


extracted_data = extract_text_from_pdf(pdf_path="data/")

filtered_data = filter_to_minimal_docs(extracted_data) 
texts_chunks = text_splitter(filtered_data)

print("📄 Starting PDF loading...")

embeddings = download_embeddings()

print("✅ PDF loading completed")


pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        spec=ServerlessSpec(cloud="aws",region="us-east-1"),
        metric="cosine",
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents = texts_chunks, 
    embedding = embeddings, 
    index_name=index_name
)