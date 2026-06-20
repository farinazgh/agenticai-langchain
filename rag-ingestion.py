import os

# IMPORTANT:
# Pinecone index dimension MUST match the embedding model dimension.
#
# text-embedding-3-small  -> 1536 dimensions
# text-embedding-3-large  -> 3072 dimensions
#
# If Pinecone throws:
# "Vector dimension X does not match the dimension of the index Y"
# recreate the Pinecone index with the correct dimension.
#

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
# Current setup:
# Pinecone Index Type : Dense
# Pinecone Dimension  : 1536
# Metric              : cosine

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(
        "C:\\polaris\\code\\agenticai-langchain\\rag-mediumblog.txt", encoding="utf-8"
    )

    document = loader.load()

    print("splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("ingesting...")

    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
