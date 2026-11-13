"""In-memory dataset registry enabling lightweight reference-passing in graph state."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid

import pandas as pd

# Global registry storing heavy dataset payloads out of LangGraph state
_DATA_CACHE: Dict[str, Union[pd.DataFrame, bytes, str, Path]] = {}
_LLM_CACHE: Dict[str, Any] = {}


class LLMRegistry:
    """Registry for managing custom LLM references and keeping live chat models out of graph checkpoint state."""

    @classmethod
    def register(cls, model: Any) -> Optional[str]:
        """Store custom LLM instance and return a reference token."""
        if model is None:
            return None
        token = f"llm://{uuid.uuid4().hex[:12]}"
        _LLM_CACHE[token] = model
        return token

    @classmethod
    def resolve(cls, ref: Optional[str]) -> Any:
        """Resolve an LLM reference token back to its underlying instance."""
        if not ref:
            return None
        return _LLM_CACHE.get(ref)

    @classmethod
    def clear(cls, ref: Optional[str] = None) -> None:
        """Clean up LLM registry entries."""
        if ref and ref in _LLM_CACHE:
            del _LLM_CACHE[ref]
        elif not ref:
            _LLM_CACHE.clear()


class DatasetRegistry:
    """Registry for managing dataset references and keeping raw DataFrames out of graph state."""

    @classmethod
    def register(cls, source: Any) -> str:
        """Store dataset payload and return a compact string reference token.

        If source is already a string filepath, returns it directly.
        If source is a DataFrame or bytes, caches it under a unique URI.
        """
        if isinstance(source, (str, Path)) and not str(source).startswith("registry://"):
            return str(source)

        token = f"registry://{uuid.uuid4().hex[:12]}"
        _DATA_CACHE[token] = source
        return token

    @classmethod
    def resolve(cls, ref: str) -> Any:
        """Resolve a dataset reference token back to its underlying payload."""
        if ref in _DATA_CACHE:
            return _DATA_CACHE[ref]
        return ref

    @classmethod
    def clear(cls, ref: Optional[str] = None) -> None:
        """Clean up registry entries."""
        if ref and ref in _DATA_CACHE:
            del _DATA_CACHE[ref]
        elif not ref:
            _DATA_CACHE.clear()
