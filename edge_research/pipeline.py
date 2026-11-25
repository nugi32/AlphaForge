"""
Stages 0-2 runner with per-stage fingerprints (resume = skip a stage whose fingerprint and outputs are unchanged).

Fingerprint of a stage = data hash + hash of that stage's config sections + stage code version. The manifest records the data hash,
config hash, package version, seed and holdout accesses (none yet in stages 0-2).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from edge_research import __version__
from edge_research.config import config_hash
from edge_research.atoms.atoms import build_atoms, family_map, load_atoms, save_atoms, summary_md as atoms_summary
from edge_research.discovery.candidates import build_candidates
from edge_research.discovery.pairs import coarse_screen
from edge_research.discovery.deep import deep_scope_note, expand_survivors, normalise_rules
from edge_research.discovery.filters import run_filters
from edge_research.discovery.exits import build_exits
from edge_research.discovery.trees import induce_rules
from edge_research.nulltest.null import load_distribution, null_base_fingerprint, null_summary, rule_flags, run_null, write_null
from edge_research.discovery.report import write_candidates, write_deep, write_exits, write_filters, write_screen
from edge_research.eda.univariate import run_univariate, write_eda
from edge_research.features.cluster import cluster_features
from edge_research.features.leakcheck import check_no_lookahead
from edge_research.features.table import build_feature_table, write_features
from edge_research.ingestion.loader import ingest_pipeline
from edge_research.labels.builder import build_labels, write_labels
from edge_research.qa.ingest_qa import run_qa, write_report
from edge_research.splits import InsufficientSplitDataError, make_splits
from edge_research.rules import load_rules, save_rules
from edge_research.validation import holdout as hold_mod
from edge_research.validation.freeze import freeze_rules
from edge_research.validation.gates import build_gate_table, gates_md, regime_diag

STAGE_VERSIONS = {"00_data_qa": "1", "01_features": "1", "02_labels": "1", "03_eda": "1", "04_atoms": "1", "05_discovery": "1", "05b_deep": "1",
                  "06_candidates": "1", "07_filters": "1", "08_exits": "1", "09_null": "1", "10_validation": "1", "11_backtest": "1", "12_ea": "1", "13_report": "1"}
STAGE_SECTIONS = {"00_data_qa": ["qa", "data"], "01_features": ["features", "data"], "02_labels": ["labels", "costs", "data"],
                  "03_eda": ["eda", "features", "labels", "costs", "data"],
                  "04_atoms": ["atoms", "features", "labels", "data"],
                  "05_discovery": ["discovery", "atoms", "features", "labels", "costs", "data"],
                  "05b_deep": ["deep", "discovery", "atoms", "features", "labels", "costs", "data"],
                  "06_candidates": ["candidates", "deep", "discovery", "atoms", "features", "labels", "costs", "data"],
                  "07_filters": ["filters", "candidates", "deep", "discovery", "atoms", "features", "labels", "costs", "data"],
                  "08_exits": ["exits", "filters", "candidates", "deep", "discovery", "atoms", "features", "labels", "costs", "data"],
                  "09_null": ["nulltest", "deep", "filters", "candidates", "discovery", "atoms", "features", "labels", "costs", "data"],   # labels: the embargo (hence the train range) depends on the hold lengths
                  "10_validation": ["validation", "engine", "exits", "filters", "candidates", "deep", "discovery", "atoms", "features", "labels", "costs", "data"],
                  "11_backtest": ["engine", "backtest", "validation", "costs", "data"],
                  "12_ea": ["codegen", "engine", "features", "costs", "data"],
                  "13_report": ["reporting", "validation", "engine", "costs", "data"]}

logger = logging.getLogger(__name__)


class QAFailed(RuntimeError):
    pass


class NoGo(RuntimeError):
    """Raised after Stage 3 outputs are written when the verdict is NO_GO and ``eda.stop_on_no_go`` is true."""


def _is_loader_mode(cfg: Dict[str, Any]) -> bool:
    return cfg.get("data", {}).get("source") == "mql5_loader"


def load_spread_csv(path: str | Path) -> pd.Series:
    raw = pd.read_csv(path)
    t = pd.to_datetime(raw["time"], format="%Y%m%d %H:%M")
    return pd.Series(raw["spread"].to_numpy("float32"), index=pd.DatetimeIndex(t, name="time"), name="spread")


def _fingerprint(stage: str, data_hash: str, cfg: Dict[str, Any]) -> str:
    parts = [data_hash, STAGE_VERSIONS[stage]] + [config_hash(cfg, s) for s in STAGE_SECTIONS[stage]]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _load_manifest(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"stages": {}, "holdout_accesses": []}


def run_stages_0_2(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    status: Dict[str, str] = {}

    df, _ = ingest_pipeline(csv_path, run / "_ingest", symbol, cfg["data"]["timeframe"])      # unchanged loader
    spread = load_spread_csv(spread_csv) if spread_csv else None
    if _is_loader_mode(cfg):                                       # MQL5 loader mode: ONE csv; envelope repair, integrity checks, spread from the data table, coverage guard
        from edge_research.mql5_loader.integration import prepare_run_inputs
        df, spread = prepare_run_inputs(df, cfg, spread, run / "_ingest")

    def done(stage: str, fp: str, marker: str) -> bool:
        return manifest["stages"].get(stage, {}).get("fingerprint") == fp and (run / stage / marker).exists()

    def mark(stage: str, fp: str) -> None:
        manifest["stages"][stage] = {"fingerprint": fp, "status": "done"}
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- stage 0 -------------------------------------------------------------------------------------------------
    import edge_research.ingestion.loader as _loader
    _loader._TIME_MIN_PARSE_FRACTION = float(cfg["qa"]["min_time_parse_fraction"])    # the loader is verbatim (NOTES D-002): the threshold is set here
    rep = run_qa(df, cfg, spread)
    manifest.update({"package_version": __version__, "seed": cfg["seed"], "data_hash": rep.data_hash, "config_hash": config_hash(cfg)})
    fp = _fingerprint("00_data_qa", rep.data_hash, cfg)
    if not done("00_data_qa", fp, "report.md"):
        write_report(rep, run / "00_data_qa")
        mark("00_data_qa", fp)
        status["00_data_qa"] = "ran"
    else:
        status["00_data_qa"] = "skipped"
    if rep.critical_failed:
        raise QAFailed("critical QA check failed; see " + str(run / "00_data_qa" / "report.md"))

    # ---- stage 1 -------------------------------------------------------------------------------------------------
    fp = _fingerprint("01_features", rep.data_hash, cfg)
    feat = build_feature_table(df, cfg, exclude_sources=set(rep.excluded_columns))
    if not done("01_features", fp, "feature_table.parquet"):
        write_features(feat, run / "01_features")
        try:
            splits = make_splits(len(df), cfg)
        except InsufficientSplitDataError as exc:
            logger.warning(
                "Skipping feature leakcheck and correlation clustering: %s. "
                "All built features are retained without correlation pruning; later stages that need "
                "train/validation/holdout splits still require more rows.",
                exc,
            )
            (run / "01_features" / "leakcheck.json").write_text(
                json.dumps({"status": "skipped", "reason": str(exc)}), encoding="utf-8"
            )
            kept = [spec.name for spec in feat.specs]
            cmap = pd.DataFrame(
                [
                    {"cluster": i, "feature": name, "representative": True, "rho_to_rep": 1.0}
                    for i, name in enumerate(kept, start=1)
                ],
                columns=["cluster", "feature", "representative", "rho_to_rep"],
            )
        else:
            lc_cfg = cfg["features"].get("leakcheck", "auto")                # spec 6.5 acceptance: automatic look-ahead test for every feature
            if lc_cfg is True or (lc_cfg == "auto" and len(df) <= 200_000):
                cut = max(splits.train.start, feat.first_valid_row) + max(1, (splits.train.stop - max(splits.train.start, feat.first_valid_row)) // 2)
                leaked = check_no_lookahead(lambda d: build_feature_table(d, cfg, exclude_sources=set(rep.excluded_columns)).table, df, cut)
                (run / "01_features" / "leakcheck.json").write_text(json.dumps({"cut": cut, "leaked_features": leaked}), encoding="utf-8")
                if leaked:
                    raise QAFailed(f"look-ahead detected in features {leaked[:8]}; see 01_features/leakcheck.json")
            train = feat.table.iloc[max(splits.train.start, feat.first_valid_row) : splits.train.stop]
            kept, cmap = cluster_features(train, feat.specs, cfg["features"]["corr_cluster_threshold"])
        cmap.to_csv(run / "01_features" / "family_clusters.csv", index=False)
        (run / "01_features" / "kept_features.json").write_text(json.dumps(kept), encoding="utf-8")
        if _is_loader_mode(cfg):
            from edge_research.mql5_loader.coverage import write_run_coverage
            write_run_coverage(run, cfg, ["time"] + list(df.columns))
        mark("01_features", fp)
        status["01_features"] = "ran"
    else:
        status["01_features"] = "skipped"

    # ---- stage 2 -------------------------------------------------------------------------------------------------
    fp = _fingerprint("02_labels", rep.data_hash, cfg)
    if not done("02_labels", fp, "label_table.parquet"):
        write_labels(build_labels(df, cfg, spread), run / "02_labels")
        mark("02_labels", fp)
        status["02_labels"] = "ran"
    else:
        status["02_labels"] = "skipped"
    return {"status": status, "n_rows": len(df), "n_features": feat.table.shape[1], "first_valid_row": feat.first_valid_row}


def run_stage_3(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 3 (univariate go/no-go) from the Stage 1 and 2 outputs in ``run_dir``. Uses TRAIN rows only."""
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    fp = _fingerprint("03_eda", manifest["data_hash"], cfg)
    stage = "03_eda"
    if manifest["stages"].get(stage, {}).get("fingerprint") != fp or not (run / stage / "verdict.json").exists():
        reg = json.loads((run / "01_features" / "feature_registry.json").read_text(encoding="utf-8"))
        kept = json.loads((run / "01_features" / "kept_features.json").read_text(encoding="utf-8"))
        feats = pd.read_parquet(run / "01_features" / "feature_table.parquet")
        labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
        res = run_univariate(feats, labels, {f["name"]: f["kind"] for f in reg["features"]}, kept, cfg, make_splits(len(feats), cfg), reg["first_valid_row"])
        write_eda(res, run / stage)
        manifest["stages"][stage] = {"fingerprint": fp, "status": "done", "verdict": res.verdict["verdict"]}
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status, verdict = "ran", res.verdict
    else:
        status, verdict = "skipped", json.loads((run / stage / "verdict.json").read_text(encoding="utf-8"))
    if verdict["verdict"] == "NO_GO" and cfg["eda"].get("stop_on_no_go", False):
        raise NoGo("Stage 3 verdict NO_GO; see " + str(run / stage / "summary.md"))
    return {"status": status, "verdict": verdict["verdict"], "n_hits": verdict["n_hits"], "n_consistent_hits": verdict["n_consistent_hits"]}


def run_stages_0_3(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    out = run_stages_0_2(csv_path, run_dir, cfg, symbol, spread_csv)
    out["status"]["03_eda"] = None
    s3 = run_stage_3(run_dir, cfg)
    out["status"]["03_eda"] = s3["status"]
    out["eda"] = {k: s3[k] for k in ("verdict", "n_hits", "n_consistent_hits")}
    return out


def run_stage_4(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 4 (atoms) from the Stage 1 outputs. Thresholds come from TRAIN rows only; masks are materialised on all rows."""
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "04_atoms"
    fp = _fingerprint(stage, manifest["data_hash"], cfg)
    if manifest["stages"].get(stage, {}).get("fingerprint") == fp and (run / stage / "atoms.json").exists():
        meta = json.loads((run / stage / "atoms.json").read_text(encoding="utf-8"))["meta"]
        return {"status": "skipped", "n_atoms": meta["n_atoms"]}
    reg = json.loads((run / "01_features" / "feature_registry.json").read_text(encoding="utf-8"))
    kept = json.loads((run / "01_features" / "kept_features.json").read_text(encoding="utf-8"))
    cmap = pd.read_csv(run / "01_features" / "family_clusters.csv")
    feats = pd.read_parquet(run / "01_features" / "feature_table.parquet")
    sp = make_splits(len(feats), cfg)
    train = slice(max(sp.train.start, int(reg["first_valid_row"])), sp.train.stop)
    a = build_atoms(feats, {f["name"]: f["kind"] for f in reg["features"]}, family_map(cmap), kept, cfg, train)
    save_atoms(a, run / stage, {"data_hash": manifest["data_hash"], "stage_fingerprint": fp, "seed": cfg["seed"]})
    (run / stage / "summary.md").write_text(atoms_summary(a), encoding="utf-8")
    manifest["stages"][stage] = {"fingerprint": fp, "status": "done", "n_atoms": a.meta["n_atoms"]}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran", "n_atoms": a.meta["n_atoms"]}


def run_stages_0_4(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    out = run_stages_0_3(csv_path, run_dir, cfg, symbol, spread_csv)      # raises NoGo before atoms when eda.stop_on_no_go is true
    s4 = run_stage_4(run_dir, cfg)
    out["status"]["04_atoms"] = s4["status"]
    out["n_atoms"] = s4["n_atoms"]
    return out


def run_stage_5(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 5, step 1 (coarse screen of all atom pairs) from the Stage 2 and 4 outputs. Reads TRAIN rows only."""
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "05_discovery"
    fp = _fingerprint(stage, manifest["data_hash"], cfg)
    if manifest["stages"].get(stage, {}).get("fingerprint") == fp and (run / stage / "screen_info.json").exists():
        info = json.loads((run / stage / "screen_info.json").read_text(encoding="utf-8"))
        return {"status": "skipped", "n_survivors": info["n_survivors"]}
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    surv, info = coarse_screen(aset, labels, cfg, slice(*aset.train))
    write_screen(surv, info, aset, run / stage)
    manifest["stages"][stage] = {"fingerprint": fp, "status": "done", "n_survivors": info["n_survivors"]}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran", "n_survivors": info["n_survivors"]}


def run_stages_0_5(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    out = run_stages_0_4(csv_path, run_dir, cfg, symbol, spread_csv)
    s5 = run_stage_5(run_dir, cfg)
    out["status"]["05_discovery"] = s5["status"]
    out["n_survivors"] = s5["n_survivors"]
    return out


class MissingUpstream(RuntimeError):
    pass


def run_stage_5b(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 5b (M11): Apriori depth 3-4 (``discovery/deep.py``) and the shallow-tree engine (``discovery/trees.py``) on the depth-2 survivors.

    TRAIN rows only. Writes ``05b_deep/survivors_deep.parquet`` (one row per deep rule: ``atoms``, ``direction``, ``score``, ``source``),
    ``deep_info.json`` and ``summary.md``. Stage 6 consumes the depth-2 survivors AND this file.
    """
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "05b_deep"
    fp, up, done = _downstream(run, manifest, stage, "05_discovery", cfg, "deep_info.json")
    if done is not None:
        return {"status": "skipped", "n_deep": done["n_deep"], "n_apriori": done["apriori"]["n_survivors"], "n_trees": done["trees"]["n_rules"]}
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    feats = pd.read_parquet(run / "01_features" / "feature_table.parquet")
    pairs = pd.read_parquet(run / "05_discovery" / "survivors_pairs.parquet")
    rows = slice(*aset.train)
    ap, ainfo = expand_survivors(aset, pairs, labels, cfg, rows)
    tree_rules, tinfo = induce_rules(aset, feats, labels, cfg, rows)
    parts = []
    if len(ap):
        parts.append(ap.assign(source="apriori"))
    if len(tree_rules):
        parts.append(tree_rules)
    deep = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["atoms", "direction", "score", "source", "n_atoms"])
    exprs = {str(i): a.expr() for i, a in enumerate(aset.atoms)}
    info = {"apriori": ainfo, "trees": tinfo, "n_deep": int(len(deep)), "null_scope": deep_scope_note(cfg),
            "by_source": {str(k): int(v) for k, v in deep["source"].value_counts().items()} if len(deep) else {}}
    write_deep(deep, info, cfg, exprs, run / stage)
    manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", "n_deep": int(len(deep))}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran", "n_deep": int(len(deep)), "n_apriori": ainfo["n_survivors"], "n_trees": tinfo["n_rules"]}


def run_stages_0_5b(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    out = run_stages_0_5(csv_path, run_dir, cfg, symbol, spread_csv)
    s = run_stage_5b(run_dir, cfg)
    out["status"]["05b_deep"] = s["status"]
    out["n_deep"] = s["n_deep"]
    out["n_apriori"], out["n_trees"] = s["n_apriori"], s["n_trees"]
    return out


def run_stage_6(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 6 (M5 + M11): exact statistics, lift over parents, minimum episodes and dedup on the Stage 5 survivors AND the Stage 5b deep rules. TRAIN rows only.

    Both inputs are normalised to the atom-list form first, so a depth-2 pair, an Apriori triple/quad and a tree leaf walk the same code path.
    Skipped only when its own fingerprint AND the fingerprints of the Stage 5 / 5b runs it consumed are unchanged.
    """
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "06_candidates"
    up05 = manifest["stages"].get("05_discovery", {}).get("fingerprint")
    if up05 is None or not (run / "05_discovery" / "survivors_pairs.parquet").exists():
        raise MissingUpstream("Stage 5 outputs not found; run stage 5 first")
    up05b = manifest["stages"].get("05b_deep", {}).get("fingerprint")
    up = f"{up05}|{up05b or 'none'}"
    fp = _fingerprint(stage, manifest["data_hash"], cfg)
    entry = manifest["stages"].get(stage, {})
    if entry.get("fingerprint") == fp and entry.get("upstream") == up and (run / stage / "candidates_info.json").exists():
        info = json.loads((run / stage / "candidates_info.json").read_text(encoding="utf-8"))
        return {"status": "skipped", "n_candidates": info["n_candidates"], "n_clusters": info["n_clusters"], "by_depth": info.get("by_depth", {})}
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    pairs = pd.read_parquet(run / "05_discovery" / "survivors_pairs.parquet")
    parts = [normalise_rules(pairs, aset, "pairs")]
    deep_path = run / "05b_deep" / "survivors_deep.parquet"
    if up05b is not None and deep_path.exists():
        parts.append(normalise_rules(pd.read_parquet(deep_path), aset, "deep"))
    rules = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    cand, fam, info = build_candidates(rules, aset, labels, cfg, slice(*aset.train))
    write_candidates(cand, fam, info, cfg, run / stage)
    manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", "n_candidates": info["n_candidates"], "n_clusters": info["n_clusters"],
                                 "by_depth": info.get("by_depth", {})}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran", "n_candidates": info["n_candidates"], "n_clusters": info["n_clusters"], "by_depth": info.get("by_depth", {})}


def run_stages_0_6(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    out = run_stages_0_5(csv_path, run_dir, cfg, symbol, spread_csv)
    s5b = run_stage_5b(run_dir, cfg)
    out["status"]["05b_deep"] = s5b["status"]
    out["n_deep"] = s5b["n_deep"]
    s6 = run_stage_6(run_dir, cfg)
    out["status"]["06_candidates"] = s6["status"]
    out["n_candidates"], out["n_clusters"] = s6["n_candidates"], s6["n_clusters"]
    out["n_deep_candidates"] = s6.get("by_depth", {})
    return out


def _downstream(run: Path, manifest: Dict[str, Any], stage: str, upstream: str, cfg: Dict[str, Any], info_file: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Fingerprint of ``stage`` and of the ``upstream`` run it consumes; returns the stored info when the stage can be skipped."""
    up = manifest["stages"].get(upstream, {}).get("fingerprint")
    if up is None:
        raise MissingUpstream(f"stage {upstream} not found; run it first")
    fp = _fingerprint(stage, manifest["data_hash"], cfg)
    entry = manifest["stages"].get(stage, {})
    if entry.get("fingerprint") == fp and entry.get("upstream") == up and (run / stage / info_file).exists():
        return fp, up, json.loads((run / stage / info_file).read_text(encoding="utf-8"))
    return fp, up, None


def run_stage_7(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 7 (M5b): F2, F7, F4 (+ F6 soft) on the Stage 6 candidates, then dedup again. Train rows only."""
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "07_filters"
    fp, up, done = _downstream(run, manifest, stage, "06_candidates", cfg, "filters_info.json")
    if done is not None:
        return {"status": "skipped", "n_passed": done["n_passed"], "n_clusters": done["n_clusters"]}
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    cand = pd.read_parquet(run / "06_candidates" / "candidates.parquet")
    fam = pd.read_parquet(run / "06_candidates" / "candidate_families.parquet")
    log, passed, info = run_filters(cand, fam, aset, labels, cfg, slice(*aset.train))
    write_filters(log, passed, info, cfg, run / stage)
    manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", "n_passed": info["n_passed"], "n_clusters": info["n_clusters"]}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran", "n_passed": info["n_passed"], "n_clusters": info["n_clusters"]}


def run_stages_0_7(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    out = run_stages_0_6(csv_path, run_dir, cfg, symbol, spread_csv)
    s7 = run_stage_7(run_dir, cfg)
    out["status"]["07_filters"] = s7["status"]
    out["n_filtered"], out["n_filtered_clusters"] = s7["n_passed"], s7["n_clusters"]
    return out


def run_stage_8(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 8 (M6): exit selection on the smoothed surface and F3, for the candidates that passed Stage 7. Train rows only (per-family means of Stage 6)."""
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "08_exits"
    fp, up, done = _downstream(run, manifest, stage, "07_filters", cfg, "exits_info.json")
    if done is not None:
        return {"status": "skipped", "n_frozen": done["n_frozen"], "n_pass_f3": done["n_pass_f3"]}
    passed = pd.read_parquet(run / "07_filters" / "candidates_filtered.parquet")
    fam = pd.read_parquet(run / "06_candidates" / "candidate_families.parquet")
    surface, chosen, info = build_exits(passed, fam, cfg)
    write_exits(surface, chosen, info, cfg, run / stage)
    manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", "n_frozen": info["n_frozen"], "n_pass_f3": info["n_pass_f3"]}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran", "n_frozen": info["n_frozen"], "n_pass_f3": info["n_pass_f3"]}


def run_stages_0_8(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    out = run_stages_0_7(csv_path, run_dir, cfg, symbol, spread_csv)
    s8 = run_stage_8(run_dir, cfg)
    out["status"]["08_exits"] = s8["status"]
    out["n_frozen"], out["n_pass_f3"] = s8["n_frozen"], s8["n_pass_f3"]
    return out


def run_stage_9(run_dir: str | Path, cfg: Dict[str, Any], max_replicates_this_call: Optional[int] = None) -> Dict[str, Any]:
    """Stage 9 (M7): null test over the whole search. Replicates are stored one by one and reused when the search itself (data, atoms, labels, discovery, candidates,
    filters) is unchanged, so changing only the null settings (more replicates, another quantile) never recomputes finished replicates. Train rows only."""
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "09_null"
    fp, up, done = _downstream(run, manifest, stage, "08_exits", cfg, "null_info.json")
    if done is not None:
        s = done["summary"]
        return {"status": "skipped", "g1_pass": s["g1_pass"], "complete": done["run"]["complete"]}
    base_fp = null_base_fingerprint(manifest["stages"]["07_filters"]["fingerprint"], cfg["seed"])
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    chosen = pd.read_parquet(run / "08_exits" / "chosen_exits.parquet")
    feats = pd.read_parquet(run / "01_features" / "feature_table.parquet")       # the deep engines inside the null (if enabled) read feature values
    out_dir = run / stage
    rows = slice(*aset.train)
    res = run_null(aset, labels, cfg, rows, out_dir, base_fp, max_replicates_this_call=max_replicates_this_call, feature_table=feats)
    dist = load_distribution(out_dir, base_fp, cfg)
    summ = null_summary(res["real"], dist, cfg)
    flags = rule_flags(chosen, summ) if len(chosen) else chosen
    write_null(summ, dist, flags, res, cfg, out_dir)
    if res["complete"]:
        manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", "g1_pass": summ["g1_pass"]}
    else:
        manifest["stages"].pop(stage, None)                          # incomplete: the next call continues, nothing is skipped
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran" if res["complete"] else "partial", "g1_pass": summ["g1_pass"], "complete": res["complete"], "n_done": res["n_done"], "n_expected": res["n_expected"]}


def run_stages_0_9(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None,
                   max_replicates_this_call: Optional[int] = None) -> Dict[str, Any]:
    out = run_stages_0_8(csv_path, run_dir, cfg, symbol, spread_csv)
    s9 = run_stage_9(run_dir, cfg, max_replicates_this_call)
    out["status"]["09_null"] = s9["status"]
    out["g1_pass"], out["null_complete"] = s9["g1_pass"], s9["complete"]
    return out


# ---------------------------------------------------------------------------------------------------------------- stage 11 building block (M9)
def load_run_inputs(run_dir: str | Path, cfg: Dict[str, Any], spread_csv: Optional[str | Path] = None):
    """(bars, feature_table, spread, splits) of a finished run: the ingested bar table, the stage 1 feature table and the splits recomputed from the same config."""
    run = Path(run_dir)
    files = sorted(list((run / "_ingest").glob("*.parquet")) + list((run / "_ingest").glob("*.feather")))      # parquet with pyarrow, feather without (loader fallback, NOTES G-004)
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one ingested bar file in {run / '_ingest'}, found {len(files)}")
    df = pd.read_parquet(files[0]) if files[0].suffix == ".parquet" else pd.read_feather(files[0])
    if "time" in df.columns:
        df = df.set_index("time")
    ft = pd.read_parquet(run / "01_features" / "feature_table.parquet")
    spread = load_spread_csv(spread_csv) if spread_csv else None
    if _is_loader_mode(cfg):                                       # same repair and the same spread source as stage 0-2
        from edge_research.mql5_loader.integration import prepare_run_inputs
        df, spread = prepare_run_inputs(df, cfg, spread, None)
    return df, ft, spread, make_splits(len(df), cfg)


def run_backtest_stage(run_dir: str | Path, cfg: Dict[str, Any], rules_path: str | Path, segment: str = "val", spread_csv: Optional[str | Path] = None,
                       out_dir: Optional[str | Path] = None, allow_holdout: bool = False):
    """Backtest the rules in a canonical rule file on ``train`` / ``val`` / ``all``. ``holdout`` is refused here: it is opened only by the validation stage, which logs it (M8)."""
    from edge_research.mt5_backtest.backtest import run_backtest, write_backtest
    from edge_research.rules import load_rules

    if segment == "holdout" and not allow_holdout:
        raise PermissionError("the holdout segment is opened only by the validation stage (it is logged in the manifest); use train, val or all")
    if segment not in ("train", "val", "holdout", "all"):
        raise ValueError(f"unknown segment {segment!r}")
    df, ft, spread, splits = load_run_inputs(run_dir, cfg, spread_csv)
    rows = None if segment == "all" else getattr(splits, segment)
    res = run_backtest(df, cfg, load_rules(rules_path), ft, spread, rows)
    out = Path(out_dir) if out_dir else Path(run_dir) / "11_backtest" / segment
    bt = cfg.get("backtest", {}) or {}
    label = " ".join(str(x) for x in (bt.get("symbol"), cfg["data"].get("timeframe")) if x)
    files = write_backtest(res, out, f"Backtest ({segment})" + (f" - {label}" if label else ""), cfg)
    res.files = files                                              # {"report": Path, "chart": Path | None}
    return res


# ---------------------------------------------------------------------------------------------------------------- stage 10 / holdout / stage 11 (M8)
def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _read_null_flags(run: Path) -> Optional[pd.DataFrame]:
    """Stage 9 ``rule_flags.csv``; None when it is missing OR empty (no rule passed the search: the file is written without a single column)."""
    p = run / "09_null" / "rule_flags.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return None


def apply_holdout_verdicts(table: pd.DataFrame, rules, manifest: Dict[str, Any], results: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Fill the ``g10_holdout`` column from the stored holdout state: only rules that were ELIGIBLE at the opening carry a verdict; a changed or newly eligible rule is BURNED."""
    t = table.copy()
    if t.empty:
        return t
    if hold_mod.holdout_state(manifest)["status"] == "untouched":
        t["g10_holdout"] = "NOT OPENED"
        return t
    eligible = [r for r in rules if bool(t.loc[t["rule_id"] == r.rule_id, "eligible_for_holdout"].iloc[0])]
    verd = hold_mod.holdout_verdict_for(manifest, eligible, results)
    t["g10_holdout"] = [verd.get(rid, "NOT ELIGIBLE") for rid in t["rule_id"]]
    return t


def run_stage_10(run_dir: str | Path, cfg: Dict[str, Any], spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    """Stage 10 (M8): freeze the Stage 8 representatives and validate them on the validation segment with the ENGINE (gates G1-G8; G9-G11 pending or holdout).

    Reads validation rows only (the holdout is never read here, except through ``run_holdout_stage`` when ``validation.open_holdout`` is true).
    """
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "10_validation"
    up08 = manifest["stages"].get("08_exits", {}).get("fingerprint")
    if up08 is None:
        raise MissingUpstream("stage 08_exits not found; run it first")
    up09 = manifest["stages"].get("09_null", {}).get("fingerprint")
    up = f"{up08}|{up09 or 'none'}"
    fp = _fingerprint(stage, manifest["data_hash"], cfg)
    entry = manifest["stages"].get(stage, {})
    info_path = run / stage / "validation_info.json"
    if entry.get("fingerprint") == fp and entry.get("upstream") == up and info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return {"status": "skipped", **{k: info[k] for k in ("n_frozen", "n_accepted", "n_eligible", "g1_pass")}}
    aset = load_atoms(run / "04_atoms")
    labels = pd.read_parquet(run / "02_labels" / "label_table.parquet")
    chosen = pd.read_parquet(run / "08_exits" / "chosen_exits.parquet")
    fam = pd.read_parquet(run / "06_candidates" / "candidate_families.parquet")
    filtered = pd.read_parquet(run / "07_filters" / "candidates_filtered.parquet")
    g1 = None
    if up09 is not None and (run / "09_null" / "null_info.json").exists():
        g1 = json.loads((run / "09_null" / "null_info.json").read_text(encoding="utf-8"))["summary"]
    g1_pass = None if g1 is None else bool(g1["g1_pass"])
    df, ft, spread, splits = load_run_inputs(run, cfg, spread_csv)
    rules = freeze_rules(chosen, aset, cfg, max_rules=cfg["validation"]["max_rules"], null_flags=_read_null_flags(run))
    out_dir = run / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    if rules:
        table = build_gate_table(rules, chosen, filtered, fam, aset, labels, df, ft, spread, splits.val, cfg, g1_pass)
        save_rules(rules, out_dir / "frozen_rules.json")
        acc_ids = set(table.loc[table["accepted"], "rule_id"])
        save_rules([r for r in rules if r.rule_id in acc_ids], out_dir / "accepted_rules.json")
    else:
        table = pd.DataFrame()
        save_rules([], out_dir / "frozen_rules.json")
        save_rules([], out_dir / "accepted_rules.json")
    manifest = _load_manifest(mpath)
    results = hold_mod.read_results(run)
    table = apply_holdout_verdicts(table, rules, manifest, results) if len(table) else table
    table.to_csv(out_dir / "gate_table.csv", index=False)
    if rules:                                                        # spec 14.6: post-selection diagnostics
        regime_diag(rules, ft, labels, splits.val, cfg).to_csv(out_dir / "regime_diag.csv", index=False)
    note = {"untouched": "Holdout: untouched.", "started": "Holdout: an access started but did not complete (burned).", "opened": "Holdout: opened once (see manifest)."}[hold_mod.holdout_state(manifest)["status"]]
    (out_dir / "gates.md").write_text(gates_md(table, cfg, g1, note), encoding="utf-8")
    info = {"n_frozen": len(rules), "n_accepted": int(table["accepted"].sum()) if len(table) else 0, "n_eligible": int(table["eligible_for_holdout"].sum()) if len(table) else 0,
            "g1_pass": g1_pass, "k_tested": len(rules), "val_rows": [splits.val.start, splits.val.stop],
            "gate_counts": {c: int(table[c].sum()) for c in ("g2_time_stability", "g3_exit_plateau", "g4_lift", "g5_oos", "g6_trades", "g7_cost_stress", "g8_quality")} if len(table) else {}}
    info_path.write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", **{k: info[k] for k in ("n_frozen", "n_accepted", "n_eligible", "g1_pass")}}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    res = {"status": "ran", **{k: info[k] for k in ("n_frozen", "n_accepted", "n_eligible", "g1_pass")}}
    if bool(cfg["validation"]["open_holdout"]) and info["n_eligible"] > 0 and hold_mod.holdout_state(manifest)["status"] == "untouched":
        res["holdout"] = run_holdout_stage(run, cfg, spread_csv)["status"]
    return res


def run_holdout_stage(run_dir: str | Path, cfg: Dict[str, Any], spread_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    """Open the holdout ONCE for the rules that are eligible after stage 10 (G1-G8). Logged in the manifest; refused when stage 10 is stale, when nothing is eligible and on every
    second opening. The eligible rules and their gate table are fixed before the holdout is read; nothing here changes a rule."""
    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "10_validation"
    entry = manifest["stages"].get(stage, {})
    if entry.get("fingerprint") != _fingerprint(stage, manifest["data_hash"], cfg) or not (run / stage / "gate_table.csv").exists():
        raise hold_mod.HoldoutNotAllowed("stage 10 is missing or stale for this configuration; run stage 10 first, then open the holdout")
    gt_path = run / stage / "gate_table.csv"
    table = pd.read_csv(gt_path)
    rules_all = load_rules(run / stage / "frozen_rules.json")
    ids = set(table.loc[table["eligible_for_holdout"].astype(bool), "rule_id"]) if len(table) else set()
    eligible = [r for r in rules_all if r.rule_id in ids]
    df, ft, spread, splits = load_run_inputs(run, cfg, spread_csv)
    res = hold_mod.open_holdout(run, cfg, eligible, df, ft, spread, splits.holdout, _sha_file(gt_path))
    manifest = _load_manifest(mpath)
    table = apply_holdout_verdicts(table, rules_all, manifest, res)
    table.to_csv(gt_path, index=False)
    g1 = None
    if (run / "09_null" / "null_info.json").exists():
        g1 = json.loads((run / "09_null" / "null_info.json").read_text(encoding="utf-8"))["summary"]
    (run / stage / "gates.md").write_text(gates_md(table, cfg, g1, "Holdout: opened once (see manifest)."), encoding="utf-8")
    return {"status": "opened", "n_rules": len(eligible), "verdicts": {rid: r["verdict"] for rid, r in res["rules"].items()}}


def run_stages_0_10(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None,
                    max_replicates_this_call: Optional[int] = None) -> Dict[str, Any]:
    out = run_stages_0_9(csv_path, run_dir, cfg, symbol, spread_csv, max_replicates_this_call)
    if not out["null_complete"]:
        out["status"]["10_validation"] = "waiting for the null test"
        return out
    s10 = run_stage_10(run_dir, cfg, spread_csv)
    out["status"]["10_validation"] = s10["status"]
    out.update({k: s10[k] for k in ("n_frozen", "n_accepted", "n_eligible")})
    if "holdout" in s10:
        out["holdout"] = s10["holdout"]
    return out


def run_stage_11(run_dir: str | Path, cfg: Dict[str, Any], spread_csv: Optional[str | Path] = None, segments=None) -> Dict[str, Any]:
    """Stage 11: engine backtest of the rules ACCEPTED by stage 10, one shared account (``11_backtest/<segment>``), default segments train + val (``backtest.segments``,
    per symbol / TF). Every segment ends with ``trades.csv``, ``equity.csv``, ``summary.json``, ``report.md`` and the equity + balance chart ``equity_balance.png``. The holdout is not part of it."""
    run = Path(run_dir)
    segments = tuple(segments or (cfg.get("backtest", {}) or {}).get("segments") or ("train", "val"))
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "11_backtest"
    up = manifest["stages"].get("10_validation", {}).get("fingerprint")
    if up is None:
        raise MissingUpstream("stage 10_validation not found; run it first")
    fp = _fingerprint(stage, manifest["data_hash"], cfg)
    entry = manifest["stages"].get(stage, {})
    if entry.get("fingerprint") == fp and entry.get("upstream") == up and entry.get("segments") == list(segments) and (run / stage / "stage11_info.json").exists():
        return {"status": "skipped", "n_rules": entry["n_rules"]}
    acc = load_rules(run / "10_validation" / "accepted_rules.json")
    (run / stage).mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Any] = {}
    if acc:
        for seg in segments:
            res = run_backtest_stage(run, cfg, run / "10_validation" / "accepted_rules.json", segment=seg, spread_csv=spread_csv)
            summaries[seg] = {k: res.summary.get(k) for k in ("total_trades", "total_net_profit", "expectancy_R", "profit_factor", "equity_drawdown_pct")}
            ch = res.files.get("chart")
            summaries[seg]["chart"] = f"{seg}/{ch.name}" if ch else None
            summaries[seg]["report"] = f"{seg}/report.md"
    (run / stage / "stage11_info.json").write_text(json.dumps({"n_rules": len(acc), "segments": list(segments), "summaries": summaries}, indent=2, default=float), encoding="utf-8")
    manifest = _load_manifest(mpath)
    manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", "n_rules": len(acc), "segments": list(segments)}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran" if acc else "no_rules", "n_rules": len(acc), "summaries": summaries}


def run_stages_0_11(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None,
                    max_replicates_this_call: Optional[int] = None) -> Dict[str, Any]:
    out = run_stages_0_10(csv_path, run_dir, cfg, symbol, spread_csv, max_replicates_this_call)
    if out["status"].get("10_validation") == "waiting for the null test":
        return out
    s11 = run_stage_11(run_dir, cfg, spread_csv)
    out["status"]["11_backtest"] = s11["status"]
    return out


# ---------------------------------------------------------------------------------------------------------------- stage 12 (M10)
def run_stage_12(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 12: MQL5 EAs for the rules accepted by stage 10 (``12_ea/ER_<rule>.mq5``). Rules the generator cannot translate exactly are listed in ``refused.json``, never approximated.
    G9 (parity) is NOT decided here: it needs the Strategy Tester (``scripts/check_parity.py``)."""
    from edge_research.codegen.stage import generate_eas

    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "12_ea"
    up = manifest["stages"].get("10_validation", {}).get("fingerprint")
    if up is None:
        raise MissingUpstream("stage 10_validation not found; run it first")
    fp = _fingerprint(stage, manifest["data_hash"], cfg)
    entry = manifest["stages"].get(stage, {})
    if entry.get("fingerprint") == fp and entry.get("upstream") == up and (run / stage / "ea_info.json").exists():
        return {"status": "skipped", "n_generated": entry["n_generated"], "n_refused": entry["n_refused"]}
    info = generate_eas(run, cfg)
    manifest = _load_manifest(mpath)
    manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", "n_generated": len(info["generated"]), "n_refused": len(info["refused"])}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran", "n_generated": len(info["generated"]), "n_refused": len(info["refused"]), "lint_issues": info["lint_issues"]}


# ---------------------------------------------------------------------------------------------------------------- stage 13 (M12)
def run_stage_13(run_dir: str | Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 13 (M12): RUN_SUMMARY.md at the run root, the gate log and one report per accepted rule (``13_report/``).

    Reads the stage outputs only; computes no statistic and changes no verdict. Runs after stage 10 at the latest
    (it reports the validation result even when stage 11 or 12 have not run yet).
    """
    from edge_research.reporting import write_report

    run = Path(run_dir)
    mpath = run / "manifest.json"
    manifest = _load_manifest(mpath)
    stage = "13_report"
    up10 = manifest["stages"].get("10_validation", {}).get("fingerprint")
    if up10 is None:
        raise MissingUpstream("stage 10_validation not found; run it first")
    up11 = manifest["stages"].get("11_backtest", {}).get("fingerprint")
    gt = run / "10_validation" / "gate_table.csv"
    # the gate table changes WITHOUT a new stage 10 fingerprint when the holdout opens (g10 column): hash it into the upstream key
    gt_sha = hashlib.sha256(gt.read_bytes()).hexdigest()[:16] if gt.exists() else "none"
    up = f"{up10}|{up11 or 'none'}|{gt_sha}"
    fp = _fingerprint(stage, manifest["data_hash"], cfg)
    entry = manifest["stages"].get(stage, {})
    if entry.get("fingerprint") == fp and entry.get("upstream") == up and (run / stage / "RUN_SUMMARY.md").exists():
        return {"status": "skipped"}
    info = write_report(run, cfg)
    manifest = _load_manifest(mpath)
    manifest["stages"][stage] = {"fingerprint": fp, "upstream": up, "status": "done", **{k: info[k] for k in ("n_rules", "gates_logged")}}
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"status": "ran", "n_rules": info["n_rules"], "gates_logged": info["gates_logged"]}


def run_stages_0_12(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None,
                    max_replicates_this_call: Optional[int] = None) -> Dict[str, Any]:
    out = run_stages_0_11(csv_path, run_dir, cfg, symbol, spread_csv, max_replicates_this_call)
    if out["status"].get("10_validation") == "waiting for the null test":
        return out
    s12 = run_stage_12(run_dir, cfg)
    out["status"]["12_ea"] = s12["status"]
    return out


def run_stages_0_13(csv_path: str | Path, run_dir: str | Path, cfg: Dict[str, Any], symbol: str = "SYMBOL", spread_csv: Optional[str | Path] = None,
                    max_replicates_this_call: Optional[int] = None) -> Dict[str, Any]:
    out = run_stages_0_12(csv_path, run_dir, cfg, symbol, spread_csv, max_replicates_this_call)
    if out["status"].get("10_validation") == "waiting for the null test":
        return out
    s13 = run_stage_13(run_dir, cfg)
    out["status"]["13_report"] = s13["status"]
    return out
