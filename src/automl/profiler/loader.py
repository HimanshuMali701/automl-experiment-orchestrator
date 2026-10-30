"""Dataset loading utilities supporting filepaths, byte streams, and in-memory DataFrames."""

import csv
import io
from pathlib import Path
from typing import BinaryIO, TextIO, Union

import pandas as pd
from automl.core.exceptions import DatasetValidationError
from automl.core.logging import get_logger

logger = get_logger("profiler.loader")


def _check_header_duplicates(header_line: str) -> None:
    """Inspect CSV header line to ensure column names are unique."""
    if not header_line.strip():
        raise DatasetValidationError("CSV file header is empty")

    # Use csv.reader so column names are not mangled (e.g. into 'col.1')
    try:
        reader = csv.reader(io.StringIO(header_line))
        columns = next(reader, [])
    except Exception as e:
        raise DatasetValidationError(f"Could not parse CSV header: {str(e)}") from e

    seen = set()
    duplicates = set()
    for col in columns:
        col_clean = str(col).strip()
        if col_clean in seen:
            duplicates.add(col_clean)
        seen.add(col_clean)

    if duplicates:
        raise DatasetValidationError(
            f"Dataset contains duplicate column names: {sorted(list(duplicates))}",
            details={"duplicate_columns": sorted(list(duplicates))},
        )


def load_dataset(
    source: Union[str, Path, bytes, BinaryIO, TextIO, pd.DataFrame],
    max_size_mb: int = 50,
) -> pd.DataFrame:
    """Load tabular dataset into a pandas DataFrame with defensive input checks.

    Args:
        source: Filepath, bytes, file buffer, or existing DataFrame.
        max_size_mb: Maximum allowed data payload in megabytes.

    Returns:
        pd.DataFrame: The loaded DataFrame without column or row mutation.

    Raises:
        DatasetValidationError: If data is empty, violates limits, or contains duplicate columns.
    """
    max_bytes = max_size_mb * 1024 * 1024

    if isinstance(source, pd.DataFrame):
        if source.columns.duplicated().any():
            dups = source.columns[source.columns.duplicated()].unique().tolist()
            raise DatasetValidationError(
                f"DataFrame contains duplicate column names: {dups}",
                details={"duplicate_columns": dups},
            )
        return source.copy()

    # If Path or string path
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DatasetValidationError(f"File not found at path: {path}")
        file_size = path.stat().st_size
        if file_size == 0:
            raise DatasetValidationError("Dataset file is completely empty (0 bytes)")
        if file_size > max_bytes:
            raise DatasetValidationError(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum limit of {max_size_mb} MB",
                details={"file_size_bytes": file_size, "max_bytes": max_bytes},
            )

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        _check_header_duplicates(first_line)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DatasetValidationError(f"Failed to parse CSV file: {str(e)}") from e

        return df

    # If bytes
    if isinstance(source, bytes):
        byte_size = len(source)
        if byte_size == 0:
            raise DatasetValidationError("Byte stream is empty (0 bytes)")
        if byte_size > max_bytes:
            raise DatasetValidationError(
                f"Byte payload ({byte_size / (1024 * 1024):.2f} MB) exceeds maximum limit of {max_size_mb} MB",
                details={"payload_size_bytes": byte_size, "max_bytes": max_bytes},
            )

        text_stream = io.StringIO(source.decode("utf-8", errors="replace"))
        first_line = text_stream.readline()
        _check_header_duplicates(first_line)
        text_stream.seek(0)

        try:
            df = pd.read_csv(text_stream)
        except Exception as e:
            raise DatasetValidationError(f"Failed to parse CSV bytes: {str(e)}") from e

        return df

    # If file-like object (BinaryIO or TextIO)
    if hasattr(source, "read"):
        content = source.read()
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        if len(content_bytes) == 0:
            raise DatasetValidationError("Uploaded file stream is empty (0 bytes)")
        if len(content_bytes) > max_bytes:
            raise DatasetValidationError(
                f"Stream content ({len(content_bytes) / (1024 * 1024):.2f} MB) exceeds maximum limit of {max_size_mb} MB"
            )

        text_stream = io.StringIO(content_bytes.decode("utf-8", errors="replace"))
        first_line = text_stream.readline()
        _check_header_duplicates(first_line)
        text_stream.seek(0)

        try:
            df = pd.read_csv(text_stream)
        except Exception as e:
            raise DatasetValidationError(f"Failed to parse CSV stream: {str(e)}") from e

        return df

    raise DatasetValidationError(f"Unsupported data source type: {type(source).__name__}")
