from __future__ import annotations

"""
Simple retrieval script against the Milvus collection configured in .env.
 
- hybrid: combine both with a weighted ranker
 
"""

import sys
from pathlib import Path
from typing import List 

from dotenv import dotenv_values
from pymilvus import MilvusClient
from pymilvus.client.abstract import AnnSearchRequest, WeightedRanker

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