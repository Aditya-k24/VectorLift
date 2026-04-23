"""
FAISS vector index for dense retrieval.

Supports three index families:
  - ``Flat``  – exact inner-product search (small corpora, highest accuracy).
  - ``IVF``   – inverted-file flat, approximate (medium-large corpora).
  - ``HNSW``  – hierarchical NSW graph, approximate (large corpora, no GPU needed).

All index types use inner-product distance (``faiss.METRIC_INNER_PRODUCT``),
which equals cosine similarity when embeddings are L2-normalised.

Requires:
    faiss-cpu>=1.7   or   faiss-gpu>=1.7
    numpy
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# Supported index type identifiers
_INDEX_TYPES = frozenset({"Flat", "IVF", "HNSW"})


class FAISSIndex:
    """Wraps a faiss index with string-keyed ID mapping and save/load support.

    Args:
        dim:        Embedding dimension.
        index_type: One of ``"Flat"``, ``"IVF"``, ``"HNSW"``.
        nlist:      Number of Voronoi cells for IVF index (default 100).
        nprobe:     Number of cells to visit at search time for IVF (default 10).
        hnsw_m:     Number of bi-directional links per HNSW node (default 32).
        use_gpu:    Move the index to GPU (requires faiss-gpu, default False).
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "IVF",
        nlist: int = 100,
        nprobe: int = 10,
        hnsw_m: int = 32,
        use_gpu: bool = False,
    ) -> None:
        if index_type not in _INDEX_TYPES:
            raise ValueError(
                f"index_type must be one of {sorted(_INDEX_TYPES)}, got '{index_type}'."
            )

        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.use_gpu = use_gpu

        self._index: Optional[faiss.Index] = None
        self._id_map: List[str] = []           # position → passage_id
        self._id_to_pos: Dict[str, int] = {}   # passage_id → position
        self._is_trained: bool = False
        self._is_populated: bool = False

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _make_index(self) -> faiss.Index:
        """Instantiate the underlying faiss index (not yet trained)."""
        if self.index_type == "Flat":
            idx = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "IVF":
            quantiser = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFFlat(quantiser, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            idx.nprobe = self.nprobe
        elif self.index_type == "HNSW":
            idx = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
            logger.info("Moved FAISS index to GPU.")

        return idx

    def build(
        self,
        embeddings: np.ndarray,
        ids: List[str],
    ) -> None:
        """Train (if required) and populate the index.

        Args:
            embeddings: Float32 array of shape ``(n, dim)``.
            ids:        Passage ID strings, one per row.

        Raises:
            ValueError: If ``embeddings`` and ``ids`` lengths differ.
        """
        if embeddings.shape[0] != len(ids):
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but ids has {len(ids)} entries."
            )

        vectors = self._to_float32(embeddings)
        self._index = self._make_index()

        # IVF requires a training pass before adding vectors
        if self.index_type == "IVF":
            if not self._index.is_trained:
                logger.info(
                    "Training IVF index (nlist=%d) on %d vectors…",
                    self.nlist,
                    len(vectors),
                )
                self._index.train(vectors)
        self._is_trained = True

        logger.info(
            "Adding %d vectors to %s index (dim=%d).", len(vectors), self.index_type, self.dim
        )
        self._index.add(vectors)

        self._id_map = list(ids)
        self._id_to_pos = {pid: i for i, pid in enumerate(ids)}
        self._is_populated = True

        logger.info(
            "FAISSIndex built: type=%s, size=%d.", self.index_type, self._index.ntotal
        )

    def add(
        self,
        embeddings: np.ndarray,
        ids: List[str],
    ) -> None:
        """Incrementally add more vectors to an already-built index.

        The IVF index must have been trained (via :meth:`build`) before
        calling this method.

        Args:
            embeddings: Float32 array of shape ``(n, dim)``.
            ids:        Passage ID strings.
        """
        if self._index is None or not self._is_populated:
            raise RuntimeError("Call build() before add().")
        vectors = self._to_float32(embeddings)
        offset = len(self._id_map)
        self._index.add(vectors)
        for i, pid in enumerate(ids):
            self._id_map.append(pid)
            self._id_to_pos[pid] = offset + i
        logger.debug("Added %d vectors; total=%d.", len(ids), self._index.ntotal)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> Tuple[List[str], List[float]]:
        """Find the top-k most similar passages to *query_embedding*.

        Args:
            query_embedding: 1-D or 2-D float32 array.  If 1-D it is
                             treated as a single query.
            top_k:           Number of nearest neighbours to return.

        Returns:
            Tuple of ``(ids, scores)`` where both lists have length
            ``min(top_k, index_size)`` and are sorted by descending score.

        Raises:
            RuntimeError: If the index has not been built.
        """
        if self._index is None or not self._is_populated:
            raise RuntimeError("FAISSIndex has not been built. Call build() first.")

        vec = self._to_float32(query_embedding)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)

        ids: List[str] = []
        scores: List[float] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:   # faiss sentinel for "not enough results"
                continue
            ids.append(self._id_map[idx])
            scores.append(float(dist))

        return ids, scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the index and ID mapping to *path* (a directory).

        Creates two files:
        - ``faiss.index``  – the faiss binary index.
        - ``metadata.pkl`` – the ID mapping and config.

        Args:
            path: Directory to write artefacts into.
        """
        if not self._is_populated:
            raise RuntimeError("Cannot save an empty index. Call build() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Move to CPU before serialising (faiss GPU index not pickle-able)
        cpu_index = (
            faiss.index_gpu_to_cpu(self._index) if self.use_gpu else self._index
        )
        faiss.write_index(cpu_index, str(path / "faiss.index"))

        meta = {
            "dim": self.dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "hnsw_m": self.hnsw_m,
            "id_map": self._id_map,
            "id_to_pos": self._id_to_pos,
        }
        with (path / "metadata.pkl").open("wb") as fh:
            pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            "FAISSIndex saved to '%s' (%d vectors).", path, cpu_index.ntotal
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        use_gpu: bool = False,
        nprobe: Optional[int] = None,
    ) -> "FAISSIndex":
        """Load a previously saved index.

        Args:
            path:    Directory containing ``faiss.index`` and ``metadata.pkl``.
            use_gpu: Move to GPU after loading.
            nprobe:  Override the stored nprobe value (IVF only).

        Returns:
            Loaded :class:`FAISSIndex`.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index directory not found: {path}")

        with (path / "metadata.pkl").open("rb") as fh:
            meta = pickle.load(fh)

        obj = cls(
            dim=meta["dim"],
            index_type=meta["index_type"],
            nlist=meta["nlist"],
            nprobe=nprobe if nprobe is not None else meta["nprobe"],
            hnsw_m=meta["hnsw_m"],
            use_gpu=use_gpu,
        )
        idx = faiss.read_index(str(path / "faiss.index"))
        if use_gpu:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)

        # Re-apply nprobe (lost during serialisation)
        if obj.index_type == "IVF":
            idx.nprobe = obj.nprobe

        obj._index = idx
        obj._id_map = meta["id_map"]
        obj._id_to_pos = meta["id_to_pos"]
        obj._is_trained = True
        obj._is_populated = True

        logger.info(
            "FAISSIndex loaded from '%s' (%d vectors, type=%s).",
            path,
            idx.ntotal,
            obj.index_type,
        )
        return obj

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_index_size(self) -> int:
        """Return the number of vectors stored in the index."""
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    @staticmethod
    def _to_float32(arr: np.ndarray) -> np.ndarray:
        """Ensure *arr* is a contiguous float32 C-array (required by faiss)."""
        return np.ascontiguousarray(arr, dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"FAISSIndex(type='{self.index_type}', dim={self.dim}, "
            f"size={self.get_index_size()}, populated={self._is_populated})"
        )
