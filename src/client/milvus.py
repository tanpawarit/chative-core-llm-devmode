"""Utilities for connecting to the Milvus vector store.

This module centralises the logic for reading connection details from the
project's ``.env`` file and for constructing a reusable ``MilvusClient``
instance.  Keeping this in a dedicated helper avoids duplicating credential
handling across different tools or scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import dotenv_values
from pymilvus import MilvusClient


@dataclass(frozen=True)
class MilvusSettings:
    """Configuration required to work with the Milvus deployment."""

    uri: str
    collection: str
    username: Optional[str]
    password: Optional[str]
    dense_metric: str
    sparse_metric: str
    sparse_drop: Optional[float]
    partition_key_field: Optional[str]
    embed_model: str

    @property
    def token(self) -> Optional[str]:
        """Return the formatted ``token`` used by ``MilvusClient``."""

        if self.username and self.password:
            return f"{self.username}:{self.password}"
        if self.password:
            return self.password
        return None


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _from_env(key: str, *, env: dict[str, str], default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is None:
        value = env.get(key, default)
    return value


def load_settings(env_path: Optional[str] = None) -> MilvusSettings:
    env = dotenv_values(env_path or ".env")

    uri = _clean(_from_env("MILVUS_ADDR", env=env))
    collection = _clean(_from_env("MILVUS_COLLECTION", env=env))
    if not uri:
        raise ValueError("MILVUS_ADDR is not set in the environment or .env file")
    if not collection:
        raise ValueError("MILVUS_COLLECTION is not set in the environment or .env file")

    username = _clean(_from_env("MILVUS_USERNAME", env=env))
    password = _clean(_from_env("MILVUS_PASSWORD", env=env))

    dense_metric = _clean(_from_env("MILVUS_DENSE_METRIC", env=env, default="COSINE")) or "COSINE"
    sparse_metric = _clean(_from_env("MILVUS_SPARSE_METRIC", env=env, default="BM25")) or "BM25"
    sparse_drop = _to_float(_from_env("MILVUS_SPARSE_DROP", env=env))
    partition_key_field = _clean(_from_env("MILVUS_PARTITION_KEY_FIELD", env=env))
    embed_model = _clean(_from_env("OPENAI_EMBED_MODEL", env=env, default="text-embedding-3-small")) or "text-embedding-3-small"

    return MilvusSettings(
        uri=uri,
        collection=collection,
        username=username,
        password=password,
        dense_metric=dense_metric,
        sparse_metric=sparse_metric,
        sparse_drop=sparse_drop,
        partition_key_field=partition_key_field,
        embed_model=embed_model,
    )


def get_settings(env_path: Optional[str] = None) -> MilvusSettings:
    """Load Milvus configuration from environment variables or ``.env``."""

    return load_settings(env_path)


def get_milvus_client(env_path: Optional[str] = None) -> MilvusClient:
    """Build a ``MilvusClient`` instance using the loaded settings."""

    settings = get_settings(env_path)
    client_kwargs = {"uri": settings.uri}
    token = settings.token
    if token:
        client_kwargs["token"] = token
    return MilvusClient(**client_kwargs)


__all__ = ["MilvusSettings", "get_milvus_client", "get_settings", "load_settings"]
