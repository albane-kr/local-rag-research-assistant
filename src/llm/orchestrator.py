from dataclasses import dataclass
from enum import Enum
from src.llm.ollama_client import OllamaClient
from src.rag.retriever import RetrievalResult


class QueryType(str, Enum):
    SUMMARY = "summary"
    COMPARE = "compare"
    ANALYZE = "analyze"
    DEFAULT = "default"


MODEL_ROUTING = {
    QueryType.SUMMARY: "gemma4",      # Fast, good summaries
    QueryType.COMPARE: "nemotron3:33b",   # Good at multi-item reasoning
    QueryType.ANALYZE: "ibm/granite4", # Detailed analysis
    QueryType.DEFAULT: "llama3.1",     # General fallback
}


@dataclass
class LLMResponse:
    answer: str
    model_used: str
    query_type: QueryType
    chunk_ids: list[str]
    resource_ids: list[str]
    versions: list[int]


def classify_query(query: str) -> QueryType:
    """Classify query type based on keywords."""
    query_lower = query.lower()

    if any(kw in query_lower for kw in ["summarize", "summary", "brief", "overview"]):
        return QueryType.SUMMARY

    if any(kw in query_lower for kw in ["compare", "difference", "vs", "versus", "vs."]):
        return QueryType.COMPARE

    if any(kw in query_lower for kw in ["analyze", "explain", "discuss", "describe"]):
        return QueryType.ANALYZE

    return QueryType.DEFAULT


def select_model(query_type: QueryType) -> str:
    """Select model based on query type."""
    return MODEL_ROUTING.get(query_type, MODEL_ROUTING[QueryType.DEFAULT])


def generate_response(
    query: str,
    prompt: str,
    chunks: list[RetrievalResult],
    ollama: OllamaClient,
) -> LLMResponse:
    """
    LLM-01/02/03: Classify query, select model, call Ollama, return response with provenance.
    """
    query_type = classify_query(query)
    model = select_model(query_type)

    answer = ollama.generate(prompt, model=model)["response"]

    return LLMResponse(
        answer=answer,
        model_used=model,
        query_type=query_type,
        chunk_ids=[c.chunk_id for c in chunks],
        resource_ids=list(set(c.resource_id for c in chunks)),
        versions=list(set(c.version for c in chunks)),
    )
