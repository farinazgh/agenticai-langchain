#!/usr/bin/env python3
"""
rag_orchestration_pipeline.py

A clean orchestration layer for a Pinecone-backed RAG system.

This script coordinates the full query-time flow:

    user question
    -> optional query rewrite
    -> optional multi-query expansion
    -> Pinecone retrieval
    -> context deduplication
    -> grounded answer generation

It assumes the indexing pipeline has already populated Pinecone.

Install:
    pip install -r requirements.txt

Environment variables:
    OPENAI_API_KEY="..."
    PINECONE_API_KEY="..."

Run:
    python rag_orchestration_pipeline.py

Examples:
    python rag_orchestration_pipeline.py \
        --question "Who won the 2023 Cricket World Cup?"

    python rag_orchestration_pipeline.py \
        --question "What happened in the 2023 Cricket World Cup final?" \
        --query-mode multi-query

    python rag_orchestration_pipeline.py \
        --question "Who won the tournament?" \
        --query-mode rewrite-and-multi-query
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class QueryMode(str, Enum):
    ORIGINAL = "original"
    REWRITE = "rewrite"
    MULTI_QUERY = "multi-query"
    REWRITE_AND_MULTI_QUERY = "rewrite-and-multi-query"


@dataclass(frozen=True)
class OrchestrationConfig:
    # Must match indexing_pipeline and generation_pipeline.
    index_name: str = "cwc-rag-index"
    namespace: str = "wikipedia-2023-cricket-world-cup"

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    question: str = "Who won the 2023 Cricket World Cup?"
    query_mode: QueryMode = QueryMode.REWRITE_AND_MULTI_QUERY

    # Retrieval controls.
    top_k_per_query: int = 4
    max_context_docs: int = 6
    max_context_chars_per_doc: int = 1500

    # Generation controls.
    temperature: float = 0.0
    max_retries: int = 2


QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite user questions into clearer search queries for RAG retrieval.

Rules:
- Keep the user's intent.
- Do not answer the question.
- Do not add facts that are not in the question.
- Return only one rewritten query.
"""


MULTI_QUERY_SYSTEM_PROMPT = """You create multiple search queries for RAG retrieval.

Rules:
- Create 3 different search queries.
- Each query should preserve the user's intent.
- Use different wording or focus.
- Do not answer the question.
- Return only the queries, one per line.
"""


ANSWER_SYSTEM_PROMPT = """You are a careful RAG orchestration assistant.

Use only the provided retrieved context to answer the user's question.

Rules:
- If the answer is present in the context, answer clearly and directly.
- If the answer is not present in the context, say: "I don't know based on the provided context."
- Do not invent facts.
- Keep the answer concise.
- Mention the retrieved chunk numbers that support the answer.
"""


ANSWER_USER_PROMPT_TEMPLATE = """Original question:
{question}

Queries used for retrieval:
{queries}

Retrieved context:
{context}

Answer:
"""


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set.\n"
            f"Set it first, for example:\n"
            f"  export {name}='your-key-here'\n"
            f"Or put it in a .env file."
        )
    return value


def create_llm(config: OrchestrationConfig) -> ChatOpenAI:
    require_env_var("OPENAI_API_KEY")
    return ChatOpenAI(
        model=config.chat_model,
        temperature=config.temperature,
        max_retries=config.max_retries,
    )


def create_embeddings(config: OrchestrationConfig) -> OpenAIEmbeddings:
    require_env_var("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=config.embedding_model)


def create_vector_store(
    config: OrchestrationConfig,
    embeddings: OpenAIEmbeddings,
) -> PineconeVectorStore:
    api_key = require_env_var("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    if not pc.has_index(config.index_name):
        raise RuntimeError(
            f"Pinecone index '{config.index_name}' does not exist.\n"
            "Run the indexing pipeline first, or change OrchestrationConfig.index_name."
        )

    index = pc.Index(config.index_name)
    return PineconeVectorStore(index=index, embedding=embeddings)


def invoke_text(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content).strip()


def rewrite_query(llm: ChatOpenAI, question: str) -> str:
    rewritten = invoke_text(
        llm=llm,
        system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
        user_prompt=question,
    )
    return rewritten.strip().strip('"')


def generate_multi_queries(llm: ChatOpenAI, question: str) -> List[str]:
    raw = invoke_text(
        llm=llm,
        system_prompt=MULTI_QUERY_SYSTEM_PROMPT,
        user_prompt=question,
    )

    queries: List[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        cleaned = cleaned.lstrip("-•0123456789. )").strip()
        if cleaned:
            queries.append(cleaned)

    return queries[:3]


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []

    for item in items:
        normalized = " ".join(item.lower().split())
        if normalized not in seen:
            seen.add(normalized)
            result.append(item)

    return result


def build_retrieval_queries(
    llm: ChatOpenAI,
    question: str,
    mode: QueryMode,
) -> List[str]:
    if mode == QueryMode.ORIGINAL:
        return [question]

    if mode == QueryMode.REWRITE:
        return unique_preserve_order([question, rewrite_query(llm, question)])

    if mode == QueryMode.MULTI_QUERY:
        return unique_preserve_order([question, *generate_multi_queries(llm, question)])

    if mode == QueryMode.REWRITE_AND_MULTI_QUERY:
        rewritten = rewrite_query(llm, question)
        expanded = generate_multi_queries(llm, rewritten)
        return unique_preserve_order([question, rewritten, *expanded])

    raise ValueError(f"Unsupported query mode: {mode}")


def doc_identity(doc: Document) -> str:
    metadata = doc.metadata or {}

    source = metadata.get("source_url") or metadata.get("source") or ""
    chunk_index = metadata.get("chunk_index")

    if source and chunk_index is not None:
        return f"{source}::{chunk_index}"

    return hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()


def retrieve_and_dedupe(
    vector_store: PineconeVectorStore,
    queries: Sequence[str],
    config: OrchestrationConfig,
) -> List[Tuple[Document, float, str]]:
    by_id: Dict[str, Tuple[Document, float, str]] = {}

    for query in queries:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=config.top_k_per_query,
            namespace=config.namespace,
        )

        for doc, score in results:
            identity = doc_identity(doc)

            # Keep the best-ranked version if the same chunk appears from multiple queries.
            if identity not in by_id or score > by_id[identity][1]:
                by_id[identity] = (doc, score, query)

    ranked = sorted(by_id.values(), key=lambda item: item[1], reverse=True)
    return ranked[: config.max_context_docs]


def format_context_doc(
    doc: Document,
    score: float,
    query_used: str,
    display_index: int,
    max_chars: int,
) -> str:
    text = doc.page_content.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    metadata = doc.metadata or {}
    source = metadata.get("source_url") or metadata.get("source") or "unknown source"
    chunk_index = metadata.get("chunk_index", "unknown")

    return (
        f"[Retrieved chunk {display_index}]\n"
        f"Matched query: {query_used}\n"
        f"Source: {source}\n"
        f"Original chunk index: {chunk_index}\n"
        f"Similarity score: {score:.4f}\n"
        f"Content:\n{text}"
    )


def build_context(
    retrieved_docs: Sequence[Tuple[Document, float, str]],
    config: OrchestrationConfig,
) -> str:
    if not retrieved_docs:
        return "No context was retrieved."

    return "\n\n---\n\n".join(
        format_context_doc(
            doc=doc,
            score=score,
            query_used=query_used,
            display_index=i,
            max_chars=config.max_context_chars_per_doc,
        )
        for i, (doc, score, query_used) in enumerate(retrieved_docs, start=1)
    )


def generate_answer(
    llm: ChatOpenAI,
    question: str,
    queries: Sequence[str],
    context: str,
) -> str:
    user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
        question=question,
        queries="\n".join(f"- {query}" for query in queries),
        context=context,
    )

    return invoke_text(
        llm=llm,
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )


def print_retrieval_debug(
    queries: Sequence[str],
    retrieved_docs: Sequence[Tuple[Document, float, str]],
    config: OrchestrationConfig,
) -> None:
    print("\n=== Retrieval Queries ===")
    for i, query in enumerate(queries, start=1):
        print(f"{i}. {query}")

    print("\n=== Retrieved Documents After Deduplication ===")
    print(f"Namespace: {config.namespace}")
    print(f"Top-k per query: {config.top_k_per_query}")
    print(f"Max context docs: {config.max_context_docs}")

    if not retrieved_docs:
        print("No documents retrieved.")
        return

    for i, (doc, score, query_used) in enumerate(retrieved_docs, start=1):
        preview = doc.page_content.replace("\n", " ").strip()
        if len(preview) > 260:
            preview = preview[:260] + "..."

        print(f"\n--- Context doc {i} ---")
        print(f"Score: {score:.4f}")
        print(f"Matched query: {query_used}")
        print(f"Metadata: {doc.metadata}")
        print(f"Preview: {preview}")


def run(config: OrchestrationConfig) -> None:
    print("\n=== RAG Orchestration Pipeline ===")
    print(f"Pinecone index:  {config.index_name}")
    print(f"Namespace:       {config.namespace}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Chat model:      {config.chat_model}")
    print(f"Query mode:      {config.query_mode.value}")
    print(f"Question:        {config.question}")

    llm = create_llm(config)
    embeddings = create_embeddings(config)
    vector_store = create_vector_store(config, embeddings)

    queries = build_retrieval_queries(
        llm=llm,
        question=config.question,
        mode=config.query_mode,
    )

    retrieved_docs = retrieve_and_dedupe(
        vector_store=vector_store,
        queries=queries,
        config=config,
    )

    print_retrieval_debug(queries, retrieved_docs, config)

    context = build_context(retrieved_docs, config)
    answer = generate_answer(
        llm=llm,
        question=config.question,
        queries=queries,
        context=context,
    )

    print("\n=== Answer ===")
    print(answer)
    print("\n=== Done ===\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Pinecone-backed RAG orchestration pipeline."
    )

    parser.add_argument(
        "--question",
        default=OrchestrationConfig.question,
        help="User question to answer.",
    )

    parser.add_argument(
        "--query-mode",
        choices=[mode.value for mode in QueryMode],
        default=OrchestrationConfig.query_mode.value,
        help="Query orchestration strategy.",
    )

    parser.add_argument(
        "--index-name",
        default=OrchestrationConfig.index_name,
        help="Pinecone index name.",
    )

    parser.add_argument(
        "--namespace",
        default=OrchestrationConfig.namespace,
        help="Pinecone namespace.",
    )

    parser.add_argument(
        "--top-k-per-query",
        type=int,
        default=OrchestrationConfig.top_k_per_query,
        help="Number of chunks retrieved per query.",
    )

    parser.add_argument(
        "--max-context-docs",
        type=int,
        default=OrchestrationConfig.max_context_docs,
        help="Maximum deduplicated chunks sent to the LLM.",
    )

    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> OrchestrationConfig:
    return OrchestrationConfig(
        question=args.question,
        query_mode=QueryMode(args.query_mode),
        index_name=args.index_name,
        namespace=args.namespace,
        top_k_per_query=args.top_k_per_query,
        max_context_docs=args.max_context_docs,
    )


if __name__ == "__main__":
    run(config_from_args(parse_args()))
