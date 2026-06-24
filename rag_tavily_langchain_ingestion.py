# Website
# ↓
# Crawl pages
#   ↓
# Extract text
#   ↓
# Create Document objects
#   ↓
# Split into chunks
#   ↓
# Embed chunks into vectors
#   ↓
# Store in Pinecone vector DB
#   ↓
# Later: use this DB for semantic search / RAG
# Tavily to crawl/extract documentation pages
# LangChain Document objects to standardize the text
# RecursiveCharacterTextSplitter to chunk large pages
# OpenAIEmbeddings to convert text chunks into vectors
# Pinecone as the cloud vector database
# asyncio to add chunks to the vector store in batches concurrently

import asyncio
import os
import ssl
import time
from typing import List

import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import Colors, log_error, log_header, log_info, log_success, log_warning
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Configure SSL context to use certifi certificates
# “Use the certificate bundle from certifi when making HTTPS requests.”
#
# This is often added because on some machines, especially Windows or corporate environments, HTTPS requests fail with certificate errors.
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

# Pinecone setup
#
# Pinecone is not a local folder like Chroma.
# Chroma stores vectors in chroma_db/
# Pinecone stores vectors in a cloud index.
#
# The Pinecone index dimension must match the embedding model dimension.
# For text-embedding-3-small, LangChain's Pinecone example uses dimension=1536.
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "langchain-docs-2025")
pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")

pc = Pinecone(api_key=pinecone_api_key)

if not pc.has_index(pinecone_index_name):
    log_info(
        f"🌲 Pinecone: Creating index '{pinecone_index_name}'",
        Colors.DARKCYAN,
    )

    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=pinecone_cloud,
            region=pinecone_region,
        ),
    )

    # Wait until Pinecone index is ready
    while not pc.describe_index(pinecone_index_name).status["ready"]:
        log_info("⏳ Pinecone: Waiting for index to be ready...")
        time.sleep(5)

    log_success(f"Pinecone: Index '{pinecone_index_name}' is ready")
else:
    log_info(
        f"🌲 Pinecone: Using existing index '{pinecone_index_name}'",
        Colors.DARKCYAN,
    )

pinecone_index = pc.Index(pinecone_index_name)

vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings,
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )
    # This is important.
    # It splits the documents into smaller groups.
    # For example, imagine:
    # Batch 1: documents 0-499
    # Batch 2: documents 500-999
    # Batch 3: documents 1000-1199
    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    # This is where the actual indexing happens.
    # aadd_documents means: asynchronously add documents

    # take chunk text
    # ↓
    # send to OpenAI embeddings model
    # ↓
    # receive vector
    # ↓
    # store vector + text + metadata in Pinecone

    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


# crawl
# convert
# split
# index
# summarize


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site
    # Best Practice: Start small. Use depth 1 or 2 first, review the results, then increase only if necessary.

    # Excessive Depth Problem
    #
    # The risk of exploring too much of a website.
    #
    # Consequences
    # Longer runtime
    # Higher cost
    # More irrelevant pages
    # Potential exponential growth in discovered pages

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects

    # Each crawled page containsboth source attribution and extracted text:
    #
    # URL
    # Raw Content: The scraped text extracted from a page.
    # Becomes the source material for chunking and indexing.
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    # Means each chunk should be around 4000 characters.
    # Not tokens. Characters.

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )

    # Means neighboring chunks overlap by 200 characters.
    #
    # Why overlap?
    #
    # Because if a meaningful explanation is split right at the boundary, overlap helps preserve context.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
