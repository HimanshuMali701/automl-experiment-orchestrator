"""Privacy-preserving cryptographic dataset fingerprinting."""

import hashlib
from pathlib import Path
from typing import Any, Union
import pandas as pd

from automl.profiler.loader import load_dataset


def compute_dataset_fingerprint(data: Union[pd.DataFrame, str, bytes, Path]) -> str:
    """Compute a deterministic, non-reversible cryptographic SHA-256 fingerprint of a dataset.

    Privacy Guard:
    Does NOT store or process raw row values, customer PII, or confidential strings.
    Instead, hashes structural schema metadata, sorted column names, dtypes, and non-null distributions.

    Args:
        data: DataFrame, file path, or bytes.

    Returns:
        str: Cryptographic fingerprint formatted as 'sha256:<16-char-hex>'.
    """
    df = load_dataset(data) if not isinstance(data, pd.DataFrame) else data

    # Construct schema signature from structural metadata only
    schema_tokens = [
        f"shape:{len(df)}x{len(df.columns)}",
    ]

    for col in sorted(df.columns):
        col_type = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        schema_tokens.append(f"{col}:{col_type}:{null_count}")

    signature_str = "|".join(schema_tokens)
    sha_hash = hashlib.sha256(signature_str.encode("utf-8")).hexdigest()
    return f"sha256:{sha_hash[:16]}"

