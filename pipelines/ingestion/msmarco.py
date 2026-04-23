"""
MS MARCO dataset handler for VectorLift.

Supports three size modes:
  - dev   : ~1 000 passages (from the dev split qrels)
  - small : 100 000 passages (stratified sample)
  - full  : 8.8 M passages   (full MS MARCO v1.1 passage corpus)

Data is loaded via HuggingFace ``datasets`` and cached to disk so subsequent
calls are instant.  The BEIR-formatted version is used for qrels
(``"BeIR/msmarco"``) which has clean relevance judgements.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VALID_MODES = {"dev", "small", "full"}
_VALID_SPLITS = {"train", "dev", "test"}


class MSMARCODataset:
    """Lazy-loading wrapper around the MS MARCO passage dataset.

    Args:
        cache_dir:  Directory for storing downloaded and processed data.
                    Defaults to ``~/.cache/vectorlift/msmarco``.
        seed:       Random seed used when sampling passages for 'small' mode.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.cache_dir = Path(
            cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "vectorlift", "msmarco")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_passages(self, mode: str = "dev") -> List[Dict[str, str]]:
        """Load MS MARCO passages for the given mode.

        Args:
            mode: 'dev' (~1 k), 'small' (100 k), or 'full' (8.8 M).

        Returns:
            List of dicts with keys: ``id``, ``text``, ``title``.
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got '{mode}'")

        cache_file = self.cache_dir / f"passages_{mode}.jsonl"
        if cache_file.exists():
            logger.info("Loading cached passages from '%s'.", cache_file)
            return self._load_jsonl(cache_file)

        logger.info("Downloading MS MARCO passages (mode=%s) …", mode)
        passages = self._download_passages(mode)
        self._save_jsonl(passages, cache_file)
        return passages

    def load_queries(self, split: str = "dev") -> Dict[str, str]:
        """Load queries for a dataset split.

        Args:
            split: 'train', 'dev', or 'test'.

        Returns:
            Dict mapping query_id (str) -> query_text (str).
        """
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got '{split}'")

        cache_file = self.cache_dir / f"queries_{split}.json"
        if cache_file.exists():
            logger.info("Loading cached queries from '%s'.", cache_file)
            with open(cache_file, encoding="utf-8") as fh:
                return json.load(fh)

        logger.info("Downloading queries (split=%s) …", split)
        queries = self._download_queries(split)
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(queries, fh, ensure_ascii=False, indent=2)
        return queries

    def load_qrels(self, split: str = "dev") -> Dict[str, Dict[str, int]]:
        """Load relevance judgements for a dataset split.

        Args:
            split: 'train' or 'dev'.

        Returns:
            Dict mapping query_id -> {doc_id: relevance_grade}.
        """
        if split not in {"train", "dev"}:
            raise ValueError(f"qrels available for 'train' and 'dev', got '{split}'")

        cache_file = self.cache_dir / f"qrels_{split}.json"
        if cache_file.exists():
            logger.info("Loading cached qrels from '%s'.", cache_file)
            with open(cache_file, encoding="utf-8") as fh:
                return json.load(fh)

        logger.info("Downloading qrels (split=%s) …", split)
        qrels = self._download_qrels(split)
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(qrels, fh, ensure_ascii=False, indent=2)
        return qrels

    def load_training_triplets(
        self, max_samples: Optional[int] = None
    ) -> List[Tuple[str, str, str]]:
        """Load (query, positive_passage, negative_passage) training triplets.

        MS MARCO training data provides query + positive passage.  A hard
        negative is sampled from the BM25 top-100 (already part of the
        ms-marco-triples file distributed by HuggingFace).

        Args:
            max_samples: Truncate to this many triplets (``None`` = all).

        Returns:
            List of (query_text, positive_text, negative_text) tuples.
        """
        suffix = f"_{max_samples}" if max_samples else "_full"
        cache_file = self.cache_dir / f"triplets{suffix}.jsonl"

        if cache_file.exists():
            logger.info("Loading cached triplets from '%s'.", cache_file)
            raw = self._load_jsonl(cache_file)
            return [(r["query"], r["positive"], r["negative"]) for r in raw]

        logger.info("Building training triplets (max_samples=%s) …", max_samples)
        triplets = self._download_triplets(max_samples)

        records = [
            {"query": q, "positive": p, "negative": n} for q, p, n in triplets
        ]
        self._save_jsonl(records, cache_file)
        return triplets

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_passages(self, mode: str) -> List[Dict[str, str]]:
        """Download passages from HuggingFace and apply mode-based filtering."""
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required. Install it with: pip install datasets"
            ) from exc

        if mode == "dev":
            # Use the BEIR version of MS MARCO which has cleaner passage IDs
            ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", trust_remote_code=True)
            qrels_dev = self.load_qrels("dev")
            relevant_doc_ids = {
                doc_id for qrel in qrels_dev.values() for doc_id in qrel
            }
            passages: List[Dict[str, str]] = []
            for row in ds:
                if str(row["_id"]) in relevant_doc_ids:
                    passages.append({
                        "id": str(row["_id"]),
                        "text": row.get("text", ""),
                        "title": row.get("title", ""),
                    })
                if len(passages) >= 5_000:
                    break
            logger.info("Loaded %d passages for dev mode.", len(passages))
            return passages

        elif mode == "small":
            ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", trust_remote_code=True)
            all_passages = [
                {
                    "id": str(row["_id"]),
                    "text": row.get("text", ""),
                    "title": row.get("title", ""),
                }
                for row in ds
            ]
            if len(all_passages) > 100_000:
                self._rng.shuffle(all_passages)
                all_passages = all_passages[:100_000]
            logger.info("Loaded %d passages for small mode.", len(all_passages))
            return all_passages

        else:  # full
            ds = load_dataset("ms_marco", "v1.1", split="train", trust_remote_code=True)
            seen: set[str] = set()
            passages = []
            for row in ds:
                for pid, text in zip(
                    row.get("passages", {}).get("passage_id", []),
                    row.get("passages", {}).get("passage_text", []),
                ):
                    pid_str = str(pid)
                    if pid_str not in seen:
                        seen.add(pid_str)
                        passages.append({
                            "id": pid_str,
                            "text": text,
                            "title": row.get("query", ""),
                        })
            logger.info("Loaded %d passages for full mode.", len(passages))
            return passages

    def _download_queries(self, split: str) -> Dict[str, str]:
        """Download queries from HuggingFace."""
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install 'datasets': pip install datasets") from exc

        try:
            # BEIR version has pre-split query files
            ds = load_dataset("BeIR/msmarco", "queries", trust_remote_code=True)
            split_name = "train" if split == "train" else "validation"
            # Fall back to the default split structure
            available = list(ds.keys())
            logger.debug("BEIR msmarco query splits available: %s", available)

            chosen_split = split_name if split_name in available else available[0]
            queries: Dict[str, str] = {
                str(row["_id"]): row["text"]
                for row in ds[chosen_split]
            }
        except Exception:
            # Fallback: ms_marco v1.1
            logger.warning("BEIR query load failed; falling back to ms_marco v1.1.")
            ds = load_dataset("ms_marco", "v1.1", split=split, trust_remote_code=True)
            queries = {}
            for row in ds:
                qid = str(row.get("query_id", ""))
                text = row.get("query", "")
                if qid and text:
                    queries[qid] = text

        logger.info("Loaded %d queries for split '%s'.", len(queries), split)
        return queries

    def _download_qrels(self, split: str) -> Dict[str, Dict[str, int]]:
        """Download qrels from HuggingFace."""
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install 'datasets': pip install datasets") from exc

        try:
            # BEIR/msmarco ships qrels directly
            ds = load_dataset("BeIR/msmarco-qrels", trust_remote_code=True)
            available_splits = list(ds.keys())
            logger.debug("BEIR qrel splits: %s", available_splits)

            chosen = split if split in available_splits else available_splits[0]
            qrels: Dict[str, Dict[str, int]] = {}
            for row in ds[chosen]:
                qid = str(row["query-id"])
                did = str(row["corpus-id"])
                score = int(row["score"])
                qrels.setdefault(qid, {})[did] = score

        except Exception:
            logger.warning("BEIR qrel load failed; falling back to ms_marco v1.1.")
            ds = load_dataset("ms_marco", "v1.1", split=split, trust_remote_code=True)
            qrels = {}
            for row in ds:
                qid = str(row.get("query_id", ""))
                if not qid:
                    continue
                for pid, is_sel in zip(
                    row.get("passages", {}).get("passage_id", []),
                    row.get("passages", {}).get("is_selected", []),
                ):
                    if is_sel:
                        qrels.setdefault(qid, {})[str(pid)] = 1

        logger.info("Loaded qrels for %d queries (split='%s').", len(qrels), split)
        return qrels

    def _download_triplets(
        self, max_samples: Optional[int]
    ) -> List[Tuple[str, str, str]]:
        """Build (query, positive, negative) training triplets."""
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install 'datasets': pip install datasets") from exc

        ds = load_dataset("ms_marco", "v1.1", split="train", trust_remote_code=True)

        triplets: List[Tuple[str, str, str]] = []
        for row in ds:
            query = row.get("query", "")
            if not query:
                continue
            passages_info = row.get("passages", {})
            texts = passages_info.get("passage_text", [])
            labels = passages_info.get("is_selected", [])
            if not texts:
                continue

            positives = [t for t, l in zip(texts, labels) if l == 1]
            negatives = [t for t, l in zip(texts, labels) if l == 0]

            if not positives or not negatives:
                continue

            pos = self._rng.choice(positives)
            neg = self._rng.choice(negatives)
            triplets.append((query, pos, neg))

            if max_samples and len(triplets) >= max_samples:
                break

        logger.info("Built %d training triplets.", len(triplets))
        return triplets

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_jsonl(records: List[Dict[str, str]], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.debug("Saved %d records to '%s'.", len(records), path)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.debug("Loaded %d records from '%s'.", len(records), path)
        return records
