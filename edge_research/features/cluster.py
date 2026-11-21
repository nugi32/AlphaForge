"""Correlation clustering to remove near-duplicate features (spec 6.4). Computed on TRAIN rows only."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from edge_research.features.registry import FeatureSpec

EXEMPT_KINDS = {"time"}       # categorical codes: Spearman between hour/session is meaningless; always kept


def cluster_features(train_table: pd.DataFrame, specs: Sequence[FeatureSpec], threshold: float = 0.9) -> Tuple[List[str], pd.DataFrame]:
    """Average-linkage clusters at |Spearman| >= threshold; keep one representative per cluster.

    Representative = lowest MQL5 complexity, then lowest NaN fraction, then name (open question Q-001).
    Returns (kept feature names in table order, cluster map DataFrame).
    """
    by_name = {s.name: s for s in specs}
    names = [c for c in train_table.columns if by_name[c].kind not in EXEMPT_KINDS]
    exempt = [c for c in train_table.columns if by_name[c].kind in EXEMPT_KINDS]
    rank = train_table[names].rank(method="average")
    rho = np.array(rank.corr(method="pearson", min_periods=50).fillna(0.0).to_numpy(), dtype=np.float64, copy=True)   # Pearson on ranks == Spearman
    np.fill_diagonal(rho, 1.0)
    dist = 1.0 - np.abs(rho)
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=1.0 - threshold, criterion="distance") if len(names) > 1 else np.ones(len(names), int)
    nan_frac = train_table[names].isna().mean()
    rows, kept = [], set(exempt)
    for cid in sorted(set(labels)):
        members = [n for n, l in zip(names, labels) if l == cid]
        rep = min(members, key=lambda n: (by_name[n].complexity, float(nan_frac[n]), n))
        kept.add(rep)
        ri = names.index(rep)
        for m in members:
            rows.append({"cluster": int(cid), "feature": m, "representative": m == rep, "rho_to_rep": float(rho[names.index(m), ri])})
    for e in exempt:
        rows.append({"cluster": -1, "feature": e, "representative": True, "rho_to_rep": 1.0})
    cmap = pd.DataFrame(rows).sort_values(["cluster", "feature"]).reset_index(drop=True)
    return [c for c in train_table.columns if c in kept], cmap
