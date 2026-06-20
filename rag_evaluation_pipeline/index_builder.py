# index.py

import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

# --- CONFIG ---
URL = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"
INDEX_NAME = "cwc-index"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSION = 1536

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

NAMESPACE = "default"


# --- LOAD ENV ---
load_dotenv()

print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


print("Checking Pinecone index...")
if not pc.has_index(INDEX_NAME):
    print(f"Creating Pinecone index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

    print("Waiting for Pinecone index to be ready...")
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)

else:
    print(f"Pinecone index already exists: {INDEX_NAME}")


print("Connecting to Pinecone index...")
index = pc.Index(INDEX_NAME)


print("Loading documents...")
loader = WebBaseLoader(URL)
documents = loader.load()


print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_documents(documents)


print("Creating embeddings...")
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)


print("Uploading documents to Pinecone...")
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=NAMESPACE,
)

vector_store.add_documents(chunks)


print("Done.")
