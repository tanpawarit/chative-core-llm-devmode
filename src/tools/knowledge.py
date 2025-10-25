"""LangChain tool for running hybrid searches against Milvus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from pymilvus.client.abstract import AnnSearchRequest, WeightedRanker

from src.client.milvus import MilvusSettings, get_milvus_client, get_settings
from src.utils import EntityInfo


def _dense_request(query: str, *, top_k: int, settings: MilvusSettings) -> AnnSearchRequest:
    embedder = OpenAIEmbeddings(model=settings.embed_model)
    vector = embedder.embed_query(query)
    return AnnSearchRequest(
        data=[vector],
        anns_field="dense_vector",
        param={"metric_type": settings.dense_metric},
        limit=top_k,
    )


def _sparse_request(query: str, *, top_k: int, settings: MilvusSettings) -> AnnSearchRequest:
    params: Dict[str, Any] = {"metric_type": settings.sparse_metric}
    if settings.sparse_drop is not None:
        params["params"] = {"drop_ratio_search": settings.sparse_drop}
    return AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param=params,
        limit=top_k,
    )


def run_hybrid_search(
    query: str,
    *,
    branch_k: int,
    weight_dense: float,
    weight_sparse: float,
    workspace_id: Optional[str],
    settings: Optional[MilvusSettings] = None,
) -> List[Dict[str, Any]]:
    """Execute the Milvus hybrid search and return the raw hits."""

    config = settings or get_settings()
    client = get_milvus_client()

    dense = _dense_request(query, top_k=branch_k, settings=config)
    sparse = _sparse_request(query, top_k=branch_k, settings=config)

    search_kwargs: Dict[str, Any] = {
        "collection_name": config.collection,
        "reqs": [sparse, dense],
        "ranker": WeightedRanker(weight_sparse, weight_dense),
        "limit": branch_k,
        "output_fields": ["id", "text"],
    }

    if config.partition_key_field:
        if "output_fields" not in search_kwargs:
            search_kwargs["output_fields"] = []
        if config.partition_key_field not in search_kwargs["output_fields"]:
            search_kwargs["output_fields"].append(config.partition_key_field)
        if workspace_id:
            search_kwargs["filter"] = f"{config.partition_key_field} == {json.dumps(workspace_id)}"
    elif workspace_id:
        # If no partition key field is configured, we surface a warning via search results.
        # The caller can choose to handle this downstream.
        pass

    results = client.hybrid_search(**search_kwargs)
    return results[0] if results else []


def _format_hits(hits: List[Dict[str, Any]], *, settings: MilvusSettings) -> List[Dict[str, Any]]:
    """Normalize Milvus hits into serializable dicts.

    ``MilvusClient.hybrid_search`` returns a ``HybridHits`` containing ``Hit``
    objects.  These are mapping-like but are not plain dicts. We convert each
    item to a dict (via ``to_dict`` when available) before extracting fields.
    """

    formatted: List[Dict[str, Any]] = []
    for raw in hits:
        # Convert pymilvus Hit -> dict; also handle already-dict cases.
        if isinstance(raw, dict):
            hit = raw
        elif hasattr(raw, "to_dict"):
            try:
                hit = raw.to_dict()
            except Exception:
                # Fallback: skip any item that cannot be materialized
                continue
        else:
            # As a last resort, attempt dict() on mapping-like objects
            try:
                hit = dict(raw)
            except Exception:
                continue

        entity = hit.get("entity") or {}
        score = hit.get("distance")
        if score is None:
            score = hit.get("score")

        metadata = {
            key: value
            for key, value in entity.items()
            if key not in {"text", "dense_vector", "sparse_vector"}
        }

        # Partition key might be present at the top level or within entity.
        if settings.partition_key_field:
            key = settings.partition_key_field
            if key in hit:
                metadata.setdefault(key, hit[key])
            elif key in entity:
                metadata.setdefault(key, entity[key])

        formatted.append(
            {
                "id": hit.get("id"),
                "score": score,
                "text": entity.get("text"),
                "metadata": metadata,
            }
        )

    return formatted


class RetrieverError(RuntimeError):
    """Raised when the knowledge retriever cannot fulfil a request."""


@dataclass(frozen=True)
class KnowledgeSnippet:
    """Structured representation of a single knowledge result."""

    id: Optional[str]
    text: str
    score: Optional[float]
    metadata: Dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience accessor that falls back to metadata."""

        if key == "id":
            return self.id if self.id is not None else default
        if key == "text":
            return self.text if self.text is not None else default
        if key == "score":
            return self.score if self.score is not None else default
        return self.metadata.get(key, default)


class KnowledgeRetriever:
    """Simple wrapper around the Milvus hybrid search tool."""

    def __init__(
        self,
        *,
        default_top_k: int = 5,
        branch_k: int = 40,
        weight_dense: float = 0.7,
        weight_sparse: float = 0.3,
        min_entity_confidence: float = 0.5,
        max_entities: int = 5,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than zero")
        if branch_k <= 0:
            raise ValueError("branch_k must be greater than zero")
        if weight_dense < 0 or weight_sparse < 0:
            raise ValueError("weights must be non-negative")
        if weight_dense == 0 and weight_sparse == 0:
            raise ValueError("at least one of weight_dense or weight_sparse must be > 0")
        if min_entity_confidence < 0:
            raise ValueError("min_entity_confidence must be >= 0")
        if max_entities <= 0:
            raise ValueError("max_entities must be greater than zero")

        self.default_top_k = default_top_k
        self.branch_k = branch_k
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse
        self.min_entity_confidence = min_entity_confidence
        self.max_entities = max_entities
        self._cache: Dict[Tuple[str, int, Optional[str]], List[KnowledgeSnippet]] = {}

    def search(
        self,
        *,
        message: str,
        workspace_id: Optional[str] = None,
        top_k: Optional[int] = None,
        use_entities: bool = False,
        entities: Optional[Sequence[EntityInfo]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[KnowledgeSnippet]:
        """Run a hybrid knowledge search."""

        if not message or not message.strip():
            raise ValueError("message must be a non-empty string")

        limit = top_k or self.default_top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero")

        composed_query = self._compose_query(message.strip(), entities if use_entities else None)
        cache_key = (composed_query, limit, workspace_id)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        raw = hybrid_knowledge_search(
            query=composed_query,
            top_k=limit,
            branch_k=self.branch_k,
            weight_dense=self.weight_dense,
            weight_sparse=self.weight_sparse,
            workspace_id=workspace_id,
        )

        payload = self._load_payload(raw)
        results = payload.get("results") or []

        snippets: List[KnowledgeSnippet] = []
        for item in results:
            snippet = self._build_snippet(item)
            if snippet is not None:
                snippets.append(snippet)

        if filters:
            snippets = self._apply_filters(snippets, filters)

        self._cache[cache_key] = snippets
        return snippets

    def format_for_prompt(
        self,
        snippets: Sequence[KnowledgeSnippet],
        *,
        include_score: bool = True,
        max_chars: Optional[int] = None,
    ) -> str:
        """Render snippets into a compact string suitable for prompts."""

        lines: List[str] = []
        remaining = max_chars

        for index, snippet in enumerate(snippets, start=1):
            header_parts = [f"[{index}]"]
            doc_name = snippet.metadata.get("doc_name") or snippet.metadata.get("source")
            if doc_name:
                header_parts.append(str(doc_name))
            if include_score and snippet.score is not None:
                header_parts.append(f"score={snippet.score:.3f}")

            header = " ".join(header_parts)
            body = snippet.text.strip()
            block = f"{header}\n{body}"

            if remaining is not None and remaining <= 0:
                break

            if remaining is not None:
                if len(block) > remaining:
                    block = block[: max(0, remaining - 1)].rstrip() + "…"
                remaining -= len(block)

            lines.append(block)

        return "\n\n".join(lines)

    def _compose_query(
        self,
        message: str,
        entities: Optional[Sequence[EntityInfo]],
    ) -> str:
        if not entities:
            return message

        candidates = self._select_entity_values(entities)
        if not candidates:
            return message

        entity_fragment = " ".join(candidates)
        return f"{message}\n\n{entity_fragment}"

    def _select_entity_values(self, entities: Sequence[EntityInfo]) -> List[str]:
        seen: set[str] = set()
        selected: List[str] = []

        for entity in sorted(entities, key=lambda item: item.Confidence, reverse=True):
            if entity.Confidence < self.min_entity_confidence:
                continue

            value = entity.NormalizedValue or entity.OriginalValue
            if not value:
                continue

            key = value.strip().lower()
            if not key or key in seen:
                continue

            seen.add(key)
            selected.append(value.strip())
            if len(selected) >= self.max_entities:
                break

        return selected

    @staticmethod
    def _load_payload(raw: str) -> Dict[str, Any]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RetrieverError("failed to parse knowledge search payload") from exc

        if not isinstance(payload, dict):
            raise RetrieverError("knowledge search payload is not a JSON object")
        return payload

    @staticmethod
    def _apply_filters(
        snippets: Iterable[KnowledgeSnippet],
        filters: Dict[str, Any],
    ) -> List[KnowledgeSnippet]:
        def _matches(snippet: KnowledgeSnippet) -> bool:
            for key, expected in filters.items():
                actual = snippet.metadata.get(key)
                if expected is None:
                    continue
                if actual != expected:
                    return False
            return True

        return [snippet for snippet in snippets if _matches(snippet)]

    @staticmethod
    def _build_snippet(item: Dict[str, Any]) -> Optional[KnowledgeSnippet]:
        if not isinstance(item, dict):
            return None

        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            return None

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        score = item.get("score")
        if isinstance(score, (int, float)):
            score_value: Optional[float] = float(score)
        else:
            score_value = None

        identifier = item.get("id")
        if identifier is not None and not isinstance(identifier, str):
            identifier = str(identifier)

        return KnowledgeSnippet(
            id=identifier,
            text=text.strip(),
            score=score_value,
            metadata=metadata,
        )
 
def hybrid_knowledge_search(
    query: str,
    top_k: int = 5,
    branch_k: int = 40,
    weight_dense: float = 0.7,
    weight_sparse: float = 0.3,
    workspace_id: Optional[str] = "workspace_1",
) -> str:
    """Search the knowledge Milvus collection using hybrid semantic + sparse ranking.

    Args:
        query: Natural language search string.
        top_k: Maximum number of matches to return. Defaults to 5.
        branch_k: Candidate count requested from each retriever before reranking.
            Defaults to 40 and must be >= top_k.
        weight_dense: Weight for the dense (semantic) retriever component. Defaults to 0.7.
        weight_sparse: Weight for the sparse (keyword) retriever component. Defaults to 0.3.
        workspace_id: Optional workspace/tenant filter when the collection uses a
            partition key field.

    Returns:
        JSON string containing the structured search results.
    """

    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if branch_k <= 0:
        raise ValueError("branch_k must be greater than zero")

    if weight_dense < 0 or weight_sparse < 0:
        raise ValueError("weights must be non-negative")

    if weight_dense == 0 and weight_sparse == 0:
        raise ValueError("at least one of weight_dense or weight_sparse must be > 0")

    adjusted_branch_k = max(branch_k, top_k)

    # Milvus handles only modest limits efficiently; cap the candidate request to 50.
    effective_branch_k = min(adjusted_branch_k, 50)

    settings = get_settings()

    hits = run_hybrid_search(
        query.strip(),
        branch_k=effective_branch_k,
        weight_dense=weight_dense,
        weight_sparse=weight_sparse,
        workspace_id=workspace_id,
        settings=settings,
    )

    formatted = _format_hits(hits, settings=settings)
    top_results = formatted[:top_k]
    payload = {
        "query": query,
        "returned": len(top_results),
        "limit": top_k,
        "branch_k": effective_branch_k,
        "weights": {"dense": weight_dense, "sparse": weight_sparse},
        "workspace_id": workspace_id,
        "results": top_results,
    }

    return json.dumps(payload, ensure_ascii=False)

# ---- LangChain Tool wrapper -------------------------------------------------
class KnowledgeSearchInput(BaseModel):
    """Input schema for the `knowledge_search` tool."""

    message: str = Field(..., description="Natural language query to search the knowledge base.")


@tool("knowledge_search", args_schema=KnowledgeSearchInput)
def knowledge_search_tool(message: str) -> str:
    """Search the knowledge base (Milvus hybrid) and return a concise, prompt-ready string.
    """

    retriever = KnowledgeRetriever(default_top_k=5)
    snippets = retriever.search(message=message)
    return retriever.format_for_prompt(snippets)

__all__ = [
    "hybrid_knowledge_search",
    "run_hybrid_search",
    "KnowledgeRetriever",
    "KnowledgeSnippet",
    "RetrieverError",
    "knowledge_search_tool",
]

# Example usage:
# from src.tools.knowledge import KnowledgeRetriever
# retriever = KnowledgeRetriever()
# res = retriever.search(message="What is MBTI?") 
# retriever.format_for_prompt(res)
