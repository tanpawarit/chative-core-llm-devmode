from __future__ import annotations

"""
Simple retrieval script against the Milvus collection configured in .env.

Supports three modes:
  - dense: semantic search with the OpenAI embedding of the query
  - sparse: BM25 keyword search over the `text` field
  - hybrid: combine both with a weighted ranker

Usage examples:
  python retrieve.py "financial chart design" hybrid
  python retrieve.py "monte carlo runs" sparse
  python retrieve.py "trading setup" dense
"""

import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import dotenv_values
from pymilvus import MilvusClient
from pymilvus.client.abstract import AnnSearchRequest, WeightedRanker

from src.shared.config import EmbeddingSettings
from src.knowledge_embedding.infrastructure.openai_client import OpenAIEmbedder


def build_client_from_env() -> MilvusClient:
    env = dotenv_values(".env")
    uri = env.get("MILVUS_ADDR") or ""
    user = env.get("MILVUS_USERNAME") or ""
    password = env.get("MILVUS_PASSWORD") or ""
    kwargs = {"uri": uri}
    if user or password:
        token = f"{user}:{password}" if user else password
        kwargs["token"] = token
    return MilvusClient(**kwargs)


def dense_request(query: str, topk: int, *, settings: EmbeddingSettings) -> AnnSearchRequest:
    embedder = OpenAIEmbedder(settings)
    vec = embedder.embed_batch([query])[0].tolist()
    return AnnSearchRequest(
        data=[vec],
        anns_field="dense_vector",
        param={"metric_type": settings.milvus.dense_metric},
        limit=topk,
    )


def sparse_request(query: str, topk: int, *, metric: str = "IP") -> AnnSearchRequest:
    # Milvus accepts raw text for sparse queries; metric is controlled by the collection config.
    return AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": metric},
        limit=topk,
    )


def run_dense(
    client: MilvusClient,
    collection: str,
    q: str,
    topk: int,
    *,
    settings: EmbeddingSettings,
) -> List[dict]:
    req = dense_request(q, topk, settings=settings)
    res = client.search(
        collection_name=collection,
        data=req.data,
        anns_field=req.anns_field,
        limit=topk,
        search_params=req.param,
        output_fields=["id", "text"],
    )
    return res[0]


def run_sparse(
    client: MilvusClient,
    collection: str,
    q: str,
    topk: int,
    *,
    settings: EmbeddingSettings,
) -> List[dict]:
    req = sparse_request(q, topk, metric=settings.milvus.sparse_metric)
    res = client.search(
        collection_name=collection,
        data=req.data,
        anns_field=req.anns_field,
        limit=topk,
        search_params=req.param,
        output_fields=["id", "text"],
    )
    return res[0]


def run_hybrid(
    client: MilvusClient,
    collection: str,
    q: str,
    topk: int,
    w_dense: float,
    w_sparse: float,
    *,
    settings: EmbeddingSettings,
) -> List[dict]:
    req_d = dense_request(q, topk, settings=settings)
    req_s = sparse_request(q, topk, metric=settings.milvus.sparse_metric)

    ranker = WeightedRanker(w_sparse, w_dense)
    res = client.hybrid_search(
        collection_name=collection,
        reqs=[req_s, req_d],
        ranker=ranker,
        limit=topk,
        output_fields=["id", "text"],
    )
    return res[0]


def main() -> None:
    # Simple hardcoded demo values
    q = "financial chart design"
    mode = "sparse"  # choose from: dense | sparse | hybrid
    topk = 5
    w_dense = 0.5
    w_sparse = 0.5

    env = dotenv_values(".env")
    collection = env.get("MILVUS_COLLECTION") or ""
    if not collection:
        raise SystemExit("MILVUS_COLLECTION is not set in .env")

    client = build_client_from_env()

    embed_settings = EmbeddingSettings()

    if mode == "dense":
        hits = run_dense(client, collection, q, topk, settings=embed_settings)
    elif mode == "sparse":
        hits = run_sparse(client, collection, q, topk, settings=embed_settings)
    else:
        hits = run_hybrid(
            client,
            collection,
            q,
            topk,
            w_dense,
            w_sparse,
            settings=embed_settings,
        )

    for i, h in enumerate(hits, 1):
        text = (h.get("entity", {}).get("text") or "").splitlines()[0][:120]
        print(f"{i:02d}. id={h.get('id')}  score={h.get('distance'):.4f}  text={text}")


if __name__ == "__main__":
    main()
