from __future__ import annotations

import os
import pickle
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(frozen=True)
class _Entry(Generic[V]):
    expires_at: float
    value: V


class TTLCache(Generic[K, V]):
    """Tiny thread-safe TTL cache with LRU-ish eviction.

    - `ttl_s`: seconds to keep items
    - `max_entries`: maximum items to retain

    Notes:
    - Keys must be hashable
    - Values can be any Python object
    """

    def __init__(self, *, ttl_s: float, max_entries: int = 256):
        self._ttl_s = float(ttl_s)
        self._max_entries = max(1, int(max_entries))
        self._lock = Lock()
        self._store: OrderedDict[K, _Entry[V]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def backend(self) -> str:
        return "memory"

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "size": len(self)}

    def get(self, key: K) -> Optional[V]:
        now = time.time()
        with self._lock:
            ent = self._store.get(key)
            if ent is None:
                self._misses += 1
                return None
            if ent.expires_at <= now:
                self._store.pop(key, None)
                self._misses += 1
                return None
            # mark as recently used
            self._store.move_to_end(key)
            self._hits += 1
            return ent.value

    def set(self, key: K, value: V, *, ttl_s: float | None = None) -> None:
        now = time.time()
        ttl = self._ttl_s if ttl_s is None else float(ttl_s)
        expires_at = now + max(0.0, ttl)
        with self._lock:
            self._store[key] = _Entry(expires_at=expires_at, value=value)
            self._store.move_to_end(key)
            self._evict_if_needed(now)

    def _evict_if_needed(self, now: float) -> None:
        # prune expired first
        expired_keys = [k for k, e in self._store.items() if e.expires_at <= now]
        for k in expired_keys:
            self._store.pop(k, None)

        # then enforce size
        while len(self._store) > self._max_entries:
            self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        now = time.time()
        with self._lock:
            self._evict_if_needed(now)
            return len(self._store)


class SQLiteTTLCache(Generic[K, V]):
    """SQLite-backed TTL cache.

    Same `get`/`set` interface as `TTLCache`, but persists across restarts.
    Values and keys are pickled.
    """

    def __init__(
        self,
        *,
        ttl_s: float,
        max_entries: int = 256,
        db_path: str,
        namespace: str,
    ):
        self._ttl_s = float(ttl_s)
        self._max_entries = max(1, int(max_entries))
        self._db_path = db_path
        self._namespace = (namespace or "default").strip()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
              namespace TEXT NOT NULL,
              key_blob  BLOB NOT NULL,
              value_blob BLOB NOT NULL,
              expires_at REAL NOT NULL,
              last_access REAL NOT NULL,
              PRIMARY KEY(namespace, key_blob)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_exp ON cache(namespace, expires_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_lru ON cache(namespace, last_access)"
        )
        self._conn.commit()

    @property
    def backend(self) -> str:
        return "sqlite"

    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "size": len(self)}

    def get(self, key: K) -> Optional[V]:
        now = time.time()
        key_blob = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            row = self._conn.execute(
                "SELECT value_blob, expires_at FROM cache WHERE namespace=? AND key_blob=?",
                (self._namespace, key_blob),
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            value_blob, expires_at = row
            if float(expires_at) <= now:
                self._conn.execute(
                    "DELETE FROM cache WHERE namespace=? AND key_blob=?",
                    (self._namespace, key_blob),
                )
                self._conn.commit()
                self._misses += 1
                return None

            # update LRU
            self._conn.execute(
                "UPDATE cache SET last_access=? WHERE namespace=? AND key_blob=?",
                (now, self._namespace, key_blob),
            )
            self._conn.commit()
            try:
                val = pickle.loads(value_blob)
            except Exception:
                self._conn.execute(
                    "DELETE FROM cache WHERE namespace=? AND key_blob=?",
                    (self._namespace, key_blob),
                )
                self._conn.commit()
                self._misses += 1
                return None

            self._hits += 1
            return val

    def set(self, key: K, value: V, *, ttl_s: float | None = None) -> None:
        now = time.time()
        ttl = self._ttl_s if ttl_s is None else float(ttl_s)
        expires_at = now + max(0.0, ttl)
        key_blob = pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL)
        value_blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO cache(namespace, key_blob, value_blob, expires_at, last_access)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self._namespace, key_blob, value_blob, float(expires_at), float(now)),
            )
            self._prune(now)
            self._conn.commit()

    def _prune(self, now: float) -> None:
        # delete expired
        self._conn.execute(
            "DELETE FROM cache WHERE namespace=? AND expires_at<=?",
            (self._namespace, float(now)),
        )

        # enforce max entries by LRU
        row = self._conn.execute(
            "SELECT COUNT(1) FROM cache WHERE namespace=?",
            (self._namespace,),
        ).fetchone()
        count = int(row[0] if row else 0)
        if count <= self._max_entries:
            return

        to_remove = count - self._max_entries
        self._conn.execute(
            """
            DELETE FROM cache
            WHERE namespace=? AND key_blob IN (
              SELECT key_blob FROM cache
              WHERE namespace=?
              ORDER BY last_access ASC
              LIMIT ?
            )
            """,
            (self._namespace, self._namespace, int(to_remove)),
        )

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache WHERE namespace=?", (self._namespace,))
            self._conn.commit()

    def __len__(self) -> int:
        now = time.time()
        with self._lock:
            self._prune(now)
            row = self._conn.execute(
                "SELECT COUNT(1) FROM cache WHERE namespace=?",
                (self._namespace,),
            ).fetchone()
            return int(row[0] if row else 0)


def make_cache(*, ttl_s: float, max_entries: int, namespace: str):
    """Factory that selects memory vs sqlite backend via env vars.

    Env:
    - CACHE_BACKEND=memory|sqlite (default: memory)
    - CACHE_SQLITE_PATH (default: ./.cache/mzansi_cache.sqlite3)
    """

    backend = (os.environ.get("CACHE_BACKEND", "memory") or "memory").strip().lower()
    if backend == "sqlite":
        db_path = os.environ.get("CACHE_SQLITE_PATH")
        if not db_path:
            db_path = os.path.join(os.getcwd(), ".cache", "mzansi_cache.sqlite3")
        return SQLiteTTLCache(
            ttl_s=float(ttl_s),
            max_entries=int(max_entries),
            db_path=db_path,
            namespace=namespace,
        )

    return TTLCache(ttl_s=float(ttl_s), max_entries=int(max_entries))
