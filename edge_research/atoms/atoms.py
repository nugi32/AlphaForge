"""
Stage 4: atoms (spec chapter 9). An atom is one simple comparison on one feature:
``feature <= t``, ``feature > t`` (numeric), ``feature == v`` (categorical level) or ``feature == 1`` (event flag).

* Thresholds are quantiles of TRAIN rows only, rounded to ``threshold_sig_digits`` significant digits; after that they are fixed numbers
  (the EA never computes quantiles). Masks are built from the rounded number, so the stored threshold IS the rule.
* Only the ``<=`` mask of a numeric pair is stored (bit-packed). The ``>`` atom is derived: ``valid[feature] & ~mask``. Rows where the
  feature is NaN are False for both sides (NaN never satisfies a comparison, same as in MQL5), hence the per-feature ``valid`` mask.
* Atoms whose support on train rows is outside [min_support, 1 - min_support] are dropped. No label information is used.
* Combination masks are never stored; only atom masks are.
* ``compatible(a, b)`` encodes the pairing rules of 9.2 (same feature only as an interval; no two atoms of one feature family).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
QUANT_NAN = 255


@dataclass(frozen=True)
class Atom:
    id: int
    feature: str
    op: str                  # "<=", ">", "=="
    threshold: float         # quantile threshold, or the categorical level
    family: str
    kind: str                # "quantile" | "category" | "event"
    pct: Optional[int]       # quantile percent for kind == "quantile"
    store_idx: int           # row of the stored packed mask
    negate: bool             # True for ">" atoms (derived from the stored "<=" mask)
    train_support: float     # share of train rows where the atom is True

    def expr(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.6g}"

    def to_rule(self) -> Tuple[str, str, float]:
        return (self.feature, self.op, self.threshold)


def round_sig(v: float, digits: int) -> float:
    return float(f"{v:.{digits}g}")


def popcount(packed: np.ndarray) -> int:
    return int(_POPCOUNT[packed].sum())


def segment_packed(n_rows: int, rows: slice) -> np.ndarray:
    m = np.zeros(n_rows, dtype=bool)
    m[rows] = True
    return np.packbits(m)


def family_map(cluster_map: pd.DataFrame) -> Dict[str, str]:
    """Feature -> family id: the correlation cluster of Stage 1 (6.4); exempt features (cluster -1) are their own family."""
    return {r.feature: (f"cluster{int(r.cluster)}" if int(r.cluster) >= 0 else r.feature) for r in cluster_map.itertuples()}


def compatible(a: Atom, b: Atom) -> bool:
    """May two atoms appear in the same rule? (spec 9.2)"""
    if a.id == b.id:
        return False
    if a.feature == b.feature:
        if a.kind != "quantile" or b.kind != "quantile":
            return False                                   # two levels of one categorical feature are mutually exclusive
        lo, hi = (a, b) if a.op == ">" else (b, a)
        return lo.op == ">" and hi.op == "<=" and lo.threshold < hi.threshold      # interval  lo < x <= hi
    return a.family != b.family


@dataclass
class StoredSpec:
    """What is needed to rebuild one stored mask on any table."""
    feature: str
    op: str                  # "<=" or "=="
    threshold: float


@dataclass
class AtomSet:
    atoms: List[Atom]
    specs: List[StoredSpec]
    packed: np.ndarray                    # (n_stored, ceil(n_rows / 8)) uint8
    valid: Dict[str, np.ndarray]          # feature -> packed finite-value mask
    n_rows: int
    train: Tuple[int, int]                # train row range used for thresholds and support
    cuts: Dict[str, List[float]]          # numeric feature -> sorted thresholds (8-bit quantization)
    quantized: Optional[np.ndarray]       # (n_rows, n_numeric) uint8 bin index, 255 = NaN
    quantized_features: List[str]
    dropped: List[Dict[str, Any]]
    meta: Dict[str, Any]

    def packed_mask(self, atom: Atom | int) -> np.ndarray:
        a = self.atoms[atom] if isinstance(atom, int) else atom
        m = self.packed[a.store_idx]
        return (self.valid[a.feature] & ~m) if a.negate else m

    def dense(self, atom: Atom | int, rows: Optional[slice] = None) -> np.ndarray:
        d = np.unpackbits(self.packed_mask(atom), count=self.n_rows).astype(bool)
        return d if rows is None else d[rows]

    def support(self, atom: Atom | int, rows: Optional[slice] = None) -> float:
        """Share of rows (default: the train range) where the atom is True, computed with popcount on the packed mask."""
        r = slice(*self.train) if rows is None else rows
        seg = segment_packed(self.n_rows, r)
        n = popcount(seg)
        return popcount(self.packed_mask(atom) & seg) / n if n else float("nan")

    def by_feature(self, feature: str) -> List[Atom]:
        return [a for a in self.atoms if a.feature == feature]


def _stored_mask(col: np.ndarray, s: StoredSpec) -> np.ndarray:
    m = (col <= s.threshold) if s.op == "<=" else (col == s.threshold)
    return np.packbits(m)


def _quantize(table: pd.DataFrame, cuts: Dict[str, List[float]]) -> Tuple[np.ndarray, List[str]]:
    feats = list(cuts)
    q = np.full((len(table), len(feats)), QUANT_NAN, dtype=np.uint8)
    for j, f in enumerate(feats):
        x = table[f].to_numpy(np.float64)
        ok = np.isfinite(x)
        q[ok, j] = np.searchsorted(np.asarray(cuts[f]), x[ok], side="left")        # x <= cuts[k]  <=>  bin <= k
    return q, feats


def _materialize(table: pd.DataFrame, specs: List[StoredSpec], features: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    cols = {f: table[f].to_numpy(np.float64) for f in features}
    packed = np.stack([_stored_mask(cols[s.feature], s) for s in specs]) if specs else np.zeros((0, (len(table) + 7) // 8), dtype=np.uint8)
    valid = {f: np.packbits(np.isfinite(c)) for f, c in cols.items()}
    return packed, valid


def build_atoms(table: pd.DataFrame, kinds: Dict[str, str], families: Dict[str, str], kept: Sequence[str], cfg: Dict[str, Any],
                train_rows: slice, keep_quantized: bool = True) -> AtomSet:
    """Create atoms from TRAIN rows ``train_rows`` of ``table`` and materialise their masks on ALL rows of ``table``."""
    c = cfg["atoms"]
    pcts, min_sup, digits = [int(p) for p in c["quantile_pcts"]], float(c["min_support"]), int(c["threshold_sig_digits"])
    n = len(table)
    tr = table.iloc[train_rows]
    n_tr = len(tr)
    if n_tr < 100:
        raise ValueError(f"train segment too short for atoms: {n_tr} rows")

    atoms: List[Atom] = []
    specs: List[StoredSpec] = []
    dropped: List[Dict[str, Any]] = []
    cuts: Dict[str, List[float]] = {}

    def add(feature, op, thr, kind, pct, store_idx, negate, support):
        atoms.append(Atom(len(atoms), feature, op, float(thr), families.get(feature, feature), kind, pct, store_idx, negate, float(support)))

    for f in kept:
        x = tr[f].to_numpy(np.float64)
        fin = x[np.isfinite(x)]
        if len(fin) == 0:
            dropped.append({"feature": f, "what": "all", "reason": "no finite train values", "support": 0.0})
            continue
        levels = np.unique(fin)
        categorical = kinds.get(f) == "time" or len(levels) <= int(c["max_categorical_levels"])
        if categorical:
            is_flag = kinds.get(f) == "event" and set(levels.tolist()) <= {0.0, 1.0}
            for v in ([1.0] if is_flag else levels.tolist()):
                sup = float(np.sum(x == v)) / n_tr
                if v == 1.0 and is_flag and 1.0 not in levels:
                    dropped.append({"feature": f, "what": "== 1", "reason": "flag never true in train", "support": 0.0})
                    continue
                if sup < min_sup or sup > 1.0 - min_sup:
                    dropped.append({"feature": f, "what": f"== {v:g}", "reason": "support outside limits", "support": sup})
                    continue
                specs.append(StoredSpec(f, "==", float(v)))
                add(f, "==", v, "event" if is_flag else "category", None, len(specs) - 1, False, sup)
            continue
        seen: set = set()
        kept_thr: List[float] = []
        for p in pcts:
            thr = round_sig(float(np.quantile(fin, p / 100.0)), digits)
            if thr in seen:
                dropped.append({"feature": f, "what": f"p{p}", "reason": "duplicate threshold (ties)", "support": float("nan")})
                continue
            seen.add(thr)
            sup = float(np.sum(x <= thr)) / n_tr                  # NaN rows count as False, like in the stored mask
            if sup < min_sup or sup > 1.0 - min_sup:
                dropped.append({"feature": f, "what": f"p{p} ({thr:g})", "reason": "support outside limits", "support": sup})
                continue
            kept_thr.append(thr)
            specs.append(StoredSpec(f, "<=", thr))
            add(f, "<=", thr, "quantile", p, len(specs) - 1, False, sup)
            add(f, ">", thr, "quantile", p, len(specs) - 1, True, float(np.sum(np.isfinite(x) & (x > thr))) / n_tr)
        if kept_thr:
            cuts[f] = sorted(kept_thr)

    features = sorted({s.feature for s in specs})
    packed, valid = _materialize(table, specs, features)
    q, qf = _quantize(table, cuts) if keep_quantized and cuts else (None, list(cuts))
    meta = {"n_atoms": len(atoms), "n_stored_masks": len(specs), "n_features_with_atoms": len(features),
            "train_rows": n_tr, "config": {k: c[k] for k in ("quantile_pcts", "min_support", "max_categorical_levels", "threshold_sig_digits")}}
    return AtomSet(atoms, specs, packed, valid, n, (train_rows.start or 0, train_rows.stop if train_rows.stop is not None else n), cuts, q, qf, dropped, meta)


def apply_atoms(atomset: AtomSet, table: pd.DataFrame) -> AtomSet:
    """Re-materialise the FROZEN atoms (same thresholds) on another table, e.g. validation or holdout rows. No quantile is recomputed."""
    features = sorted({s.feature for s in atomset.specs})
    packed, valid = _materialize(table, atomset.specs, features)
    q, qf = _quantize(table, atomset.cuts) if atomset.cuts else (None, [])
    return AtomSet(atomset.atoms, atomset.specs, packed, valid, len(table), atomset.train, atomset.cuts, q, qf, atomset.dropped, atomset.meta)


# ---- persistence --------------------------------------------------------------------------------------------------------------

def save_atoms(a: AtomSet, out_dir: str | Path, provenance: Optional[Dict[str, Any]] = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"meta": a.meta, "provenance": provenance or {}, "n_rows": a.n_rows, "train": list(a.train), "cuts": a.cuts, "quantized_features": a.quantized_features,
           "specs": [asdict(s) for s in a.specs], "atoms": [asdict(x) for x in a.atoms], "dropped": a.dropped, "valid_features": sorted(a.valid)}
    (out / "atoms.json").write_text(json.dumps(doc, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    arrays = {"packed": a.packed, **{f"valid::{f}": v for f, v in a.valid.items()}}
    if a.quantized is not None:
        arrays["quantized"] = a.quantized
    np.savez_compressed(out / "atom_masks.npz", **arrays)
    pd.DataFrame([asdict(x) for x in a.atoms]).assign(expr=[x.expr() for x in a.atoms]).to_csv(out / "atoms.csv", index=False)
    pd.DataFrame(a.dropped).to_csv(out / "dropped_atoms.csv", index=False)


def load_atoms(out_dir: str | Path) -> AtomSet:
    out = Path(out_dir)
    doc = json.loads((out / "atoms.json").read_text(encoding="utf-8"))
    z = np.load(out / "atom_masks.npz")
    atoms = [Atom(**x) for x in doc["atoms"]]
    specs = [StoredSpec(**s) for s in doc["specs"]]
    valid = {f: z[f"valid::{f}"] for f in doc["valid_features"]}
    q = z["quantized"] if "quantized" in z.files else None
    return AtomSet(atoms, specs, z["packed"], valid, doc["n_rows"], tuple(doc["train"]), doc["cuts"], q, doc["quantized_features"], doc["dropped"], doc["meta"])


def summary_md(a: AtomSet) -> str:
    kinds = pd.Series([x.kind for x in a.atoms]).value_counts().to_dict()
    L = ["# Stage 4 atoms summary", "",
         f"- Atoms: {len(a.atoms)} (stored masks: {len(a.specs)}; `>` atoms are derived from the stored `<=` mask)",
         f"- By kind: {kinds}", f"- Features with atoms: {a.meta['n_features_with_atoms']}; train rows used: {a.meta['train_rows']}",
         f"- Mask storage: {a.packed.nbytes / 1024:.1f} KiB packed for {a.n_rows} rows", f"- Dropped: {len(a.dropped)} (see dropped_atoms.csv)", ""]
    if a.dropped:
        d = pd.DataFrame(a.dropped)
        L += ["## Dropped by reason", ""] + [f"- {r}: {n}" for r, n in d["reason"].value_counts().items()] + [""]
    L += ["Thresholds come from TRAIN rows only and are fixed numbers from here on (validation and holdout apply them via `apply_atoms`).", ""]
    return "\n".join(L)
