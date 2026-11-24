"""Chronological train / validation / holdout split with embargo (spec 4.2). No shuffling, ever."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


class InsufficientSplitDataError(ValueError):
    """Raised when the requested split sizes cannot fit in the available rows."""


@dataclass(frozen=True)
class Splits:
    train: slice
    val: slice
    holdout: slice
    embargo: int

    def sizes(self) -> Dict[str, int]:
        return {k: getattr(self, k).stop - getattr(self, k).start for k in ("train", "val", "holdout")}


def max_hold_bars(cfg: Dict[str, Any]) -> int:
    lab = cfg["labels"]
    return int(max(list(lab["time_stops"]) + list(lab["barrier_grid"]["horizon"])))


def make_splits(n_rows: int, cfg: Dict[str, Any]) -> Splits:
    """Embargo bars are dropped on BOTH sides of each boundary, so no label window of one segment touches another.

    ``data.holdout_pct == 0`` means "no holdout" (outer walk-forward windows, M8): the holdout slice is empty and the validation
    segment runs to the last row. Any other value behaves exactly as before.
    """
    d = cfg["data"]
    embargo = max_hold_bars(cfg) + int(d["embargo_margin_bars"])
    t_end = int(n_rows * d["train_pct"])
    if float(d["holdout_pct"]) == 0.0:
        s = Splits(train=slice(0, t_end - embargo), val=slice(t_end + embargo, n_rows), holdout=slice(n_rows, n_rows), embargo=embargo)
        sizes = s.sizes()
        if min(sizes["train"], sizes["val"]) <= 0:
            raise InsufficientSplitDataError(f"data too short for the split with embargo {embargo}: {sizes}")
        return s
    v_end = int(n_rows * (d["train_pct"] + d["val_pct"]))
    s = Splits(train=slice(0, t_end - embargo), val=slice(t_end + embargo, v_end - embargo), holdout=slice(v_end + embargo, n_rows), embargo=embargo)
    if min(s.sizes().values()) <= 0:
        raise InsufficientSplitDataError(f"data too short for the split with embargo {embargo}: {s.sizes()}")
    return s
