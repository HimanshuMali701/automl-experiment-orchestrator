"""In-memory dataset registry enabling lightweight reference-passing in graph state."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid

import pandas as pd

# Global registry storing heavy dataset payloads out of LangGraph state
_DATA_CACHE: Dict[str, Union[pd.DataFrame, bytes, str, Path]] = {}


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
