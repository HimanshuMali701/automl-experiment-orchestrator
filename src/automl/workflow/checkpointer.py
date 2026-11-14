"""Persistent SQLite checkpointer for LangGraph workflow state."""

from contextlib import AbstractAsyncContextManager, AbstractContextManager
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
    SerializerProtocol,
    WRITES_IDX_MAP,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from automl.core.config import get_settings
from automl.core.logging import get_logger

logger = get_logger("workflow.checkpointer")


class SqliteCheckpointSaver(
    BaseCheckpointSaver[str], AbstractContextManager, AbstractAsyncContextManager
):
    """Production-ready SQLite checkpoint saver for persistent LangGraph workflows.

    Persists checkpoints, blob channel values, and pending writes in local SQLite
    database tables, enabling workflows to pause, resume across process restarts,
    and maintain thread-safe auditability.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde or JsonPlusSerializer())
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    @classmethod
    def from_conn_string(
        cls, conn_string: str, serde: Optional[SerializerProtocol] = None
    ) -> "SqliteCheckpointSaver":
        """Instantiate saver from connection string or database file path."""
        path = conn_string.replace("sqlite:///", "").replace("sqlite://", "")
        return cls(db_path=path, serde=serde)

    def _init_db(self) -> None:
        """Create checkpointer storage tables if they do not exist."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint_type TEXT NOT NULL,
                    checkpoint_data BLOB NOT NULL,
                    metadata_type TEXT NOT NULL,
                    metadata_data BLOB NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    blob_type TEXT NOT NULL,
                    blob_data BLOB NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    write_type TEXT NOT NULL,
                    write_data BLOB NOT NULL,
                    task_path TEXT,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
                """
            )

    def __enter__(self) -> "SqliteCheckpointSaver":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def __aenter__(self) -> "SqliteCheckpointSaver":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> Dict[str, Any]:
        """Load channel blob values for specified versions."""
        channel_values: Dict[str, Any] = {}
        try:
            with self._lock:
                cursor = self._conn.cursor()
                for k, v in versions.items():
                    cursor.execute(
                        """
                        SELECT blob_type, blob_data FROM checkpoint_blobs
                        WHERE thread_id=? AND checkpoint_ns=? AND channel=? AND version=?
                        """,
                        (thread_id, checkpoint_ns, k, str(v)),
                    )
                    row = cursor.fetchone()
                    if row:
                        b_type, b_data = row
                        if b_type == "empty":
                            channel_values[k] = None
                        else:
                            channel_values[k] = self.serde.loads_typed((b_type, b_data))
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            pass
        return channel_values

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple for the given thread configuration."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self._lock:
            cursor = self._conn.cursor()
            if checkpoint_id:
                cursor.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type,
                           checkpoint_data, metadata_type, metadata_data
                    FROM checkpoints
                    WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                    """,
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                row = cursor.fetchone()
            else:
                cursor.execute(
                    """
                    SELECT checkpoint_id, parent_checkpoint_id, checkpoint_type,
                           checkpoint_data, metadata_type, metadata_data
                    FROM checkpoints
                    WHERE thread_id=? AND checkpoint_ns=?
                    ORDER BY checkpoint_id DESC LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                )
                row = cursor.fetchone()

            if not row:
                return None

            c_id, p_id, c_type, c_data, m_type, m_data = row
            cursor.execute(
                """
                SELECT task_id, idx, channel, write_type, write_data, task_path
                FROM checkpoint_writes
                WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
                ORDER BY task_id, idx
                """,
                (thread_id, checkpoint_ns, c_id),
            )
            write_rows = cursor.fetchall()
            pending_writes = [
                (t_id, ch, self.serde.loads_typed((w_type, w_data)))
                for t_id, idx, ch, w_type, w_data, path in write_rows
            ]

            checkpoint_dict: Checkpoint = self.serde.loads_typed((c_type, c_data))
            metadata_dict: CheckpointMetadata = self.serde.loads_typed((m_type, m_data))
            channel_values = self._load_blobs(
                thread_id, checkpoint_ns, checkpoint_dict["channel_versions"]
            )

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": c_id,
                    }
                },
                checkpoint={**checkpoint_dict, "channel_values": channel_values},
                metadata=metadata_dict,
                pending_writes=pending_writes,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": p_id,
                        }
                    }
                    if p_id
                    else None
                ),
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoint tuples matching the specified filters."""
        thread_id = config["configurable"]["thread_id"] if config else None
        checkpoint_ns = config["configurable"].get("checkpoint_ns") if config else None
        config_cid = get_checkpoint_id(config) if config else None
        before_cid = get_checkpoint_id(before) if before else None

        query = "SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints WHERE 1=1"
        params: List[Any] = []
        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        if checkpoint_ns:
            query += " AND checkpoint_ns = ?"
            params.append(checkpoint_ns)
        if config_cid:
            query += " AND checkpoint_id = ?"
            params.append(config_cid)
        if before_cid:
            query += " AND checkpoint_id < ?"
            params.append(before_cid)
        query += " ORDER BY checkpoint_id DESC"

        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        count = 0
        for tid, ns, cid in rows:
            tup = self.get_tuple(
                {
                    "configurable": {
                        "thread_id": tid,
                        "checkpoint_ns": ns,
                        "checkpoint_id": cid,
                    }
                }
            )
            if tup:
                if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                    continue
                yield tup
                count += 1
                if limit and count >= limit:
                    break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a new checkpoint."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        c_type, c_data = self.serde.dumps_typed(
            {k: v for k, v in checkpoint.items() if k != "channel_values"}
        )
        m_type, m_data = self.serde.dumps_typed(metadata)

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                 checkpoint_type, checkpoint_data, metadata_type, metadata_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    c_type,
                    c_data,
                    m_type,
                    m_data,
                ),
            )

            for k, v in new_versions.items():
                val = checkpoint["channel_values"].get(k)
                if val is not None:
                    b_type, b_data = self.serde.dumps_typed(val)
                else:
                    b_type, b_data = "empty", b""
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_blobs
                    (thread_id, checkpoint_ns, channel, version, blob_type, blob_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_ns, k, str(v), b_type, b_data),
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store pending writes associated with a task."""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]

        with self._lock, self._conn:
            for idx, (c, v) in enumerate(writes):
                inner_idx = WRITES_IDX_MAP.get(c, idx)
                w_type, w_data = self.serde.dumps_typed(v)
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                     channel, write_type, write_data, task_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        inner_idx,
                        c,
                        w_type,
                        w_data,
                        task_path,
                    ),
                )

    def delete_thread(self, thread_id: str) -> None:
        """Purge all checkpoints and writes for a given thread."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
            self._conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id=?", (thread_id,))
            self._conn.execute("DELETE FROM checkpoint_writes WHERE thread_id=?", (thread_id,))

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return self.get_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Any:
        for it in self.list(config, filter=filter, before=before, limit=limit):
            yield it

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)


# Global checkpointer singleton
_global_checkpointer: Optional[SqliteCheckpointSaver] = None
_checkpointer_lock = threading.Lock()


def get_default_checkpointer(
    db_path: Optional[Union[str, Path]] = None,
) -> BaseCheckpointSaver:
    """Return persistent SQLite checkpointer instance.

    Uses `artifacts/checkpoints.db` by default so checkpoints survive worker
    lifecycles and application restarts.
    """
    global _global_checkpointer
    if db_path is not None:
        return SqliteCheckpointSaver(db_path=db_path)

    with _checkpointer_lock:
        if _global_checkpointer is None:
            settings = get_settings()
            db_file = settings.artifacts_dir / "checkpoints.db"
            db_file.parent.mkdir(parents=True, exist_ok=True)
            _global_checkpointer = SqliteCheckpointSaver(db_path=db_file)
        return _global_checkpointer


def reset_default_checkpointer() -> None:
    """Reset the global checkpointer singleton (primarily for testing)."""
    global _global_checkpointer
    with _checkpointer_lock:
        _global_checkpointer = None
