from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD


DEFAULT_RELATION_COLUMNS = [
    "card1",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
]


@dataclass
class RelationMapping:
    column: str
    mapping: Dict[str, int]
    unknown_index: int
    offset: int
    width: int


class SparseGraphEmbedder:
    """
    Builds transaction-entity incidence features and compresses them via SVD.
    This is a scalable structural baseline that approximates graph-context
    embeddings when deep GNN libraries are not installed.
    """

    def __init__(
        self,
        relation_columns: Iterable[str] = DEFAULT_RELATION_COLUMNS,
        embedding_dim: int = 32,
        min_frequency: int = 2,
        random_state: int = 42,
    ) -> None:
        self.relation_columns = list(relation_columns)
        self.embedding_dim = int(embedding_dim)
        self.min_frequency = int(min_frequency)
        self.random_state = int(random_state)
        self.mappings: List[RelationMapping] = []
        self.svd: TruncatedSVD | None = None
        self.fitted = False

    def _clean(self, s: pd.Series) -> pd.Series:
        return s.fillna("__MISSING__").astype(str)

    def fit(self, df: pd.DataFrame) -> "SparseGraphEmbedder":
        self.mappings = []
        offset = 0
        for col in self.relation_columns:
            values = self._clean(df[col]) if col in df.columns else pd.Series(["__MISSING__"] * len(df))
            vc = values.value_counts()
            keep = vc[vc >= self.min_frequency].index.tolist()
            mapping = {v: i for i, v in enumerate(keep)}
            unknown_index = len(mapping)
            width = unknown_index + 1
            self.mappings.append(
                RelationMapping(
                    column=col,
                    mapping=mapping,
                    unknown_index=unknown_index,
                    offset=offset,
                    width=width,
                )
            )
            offset += width

        X = self._build_matrix(df)
        max_dim = min(self.embedding_dim, max(2, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=max_dim, random_state=self.random_state)
        self.svd.fit(X)
        self.fitted = True
        return self

    def _build_matrix(self, df: pd.DataFrame) -> csr_matrix:
        n_rows = len(df)
        blocks = []
        row_index = np.arange(n_rows)

        for m in self.mappings:
            values = self._clean(df[m.column]) if m.column in df.columns else pd.Series(["__MISSING__"] * n_rows)
            encoded = values.map(m.mapping).fillna(m.unknown_index).astype(np.int32).to_numpy()
            data = np.ones(n_rows, dtype=np.float32)
            block = csr_matrix((data, (row_index, encoded)), shape=(n_rows, m.width))
            blocks.append(block)

        return hstack(blocks, format="csr", dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or self.svd is None:
            raise RuntimeError("SparseGraphEmbedder must be fit before transform.")
        X = self._build_matrix(df)
        emb = self.svd.transform(X)
        cols = [f"graph_emb_{i:02d}" for i in range(emb.shape[1])]
        return pd.DataFrame(emb, index=df.index, columns=cols)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

