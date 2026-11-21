"""Mutation checks for M11 (Engine A depth 3-4, Engine B trees, N-atom lift / gates, stage 5b wiring). Each mutant edits ONE place of a scratch
copy of the project (/tmp/mut); the selected tests of tests/test_deep_m11.py must FAIL (mutant caught). Resumable: finished mutants are kept in
output/mut_results_m11.json. Run: python scripts/dev_mutate_m11.py [name-prefix]

NOTE: the module fixture builds a 6000-bar world once per pytest call (about 1 min on a slow machine); the -k selection keeps every run to the
few tests that pin the mutated behaviour. The stage5b mutants (P*) run the end-to-end test and are the slowest ones."""
import json, os, shutil, subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
DST = Path("/tmp/mut")
RES = SRC / "output" / "mut_results_m11.json"
D = "edge_research/discovery/deep.py"
T = "edge_research/discovery/trees.py"
G = "edge_research/validation/gates.py"
C = "edge_research/discovery/candidates.py"
P = "edge_research/pipeline.py"
TST = "tests/test_deep_m11.py"

M = [
    # name, file, old, new, -k
    ("D01_rule_key_unsorted", D, 'return ",".join(str(int(i)) for i in sorted(int(x) for x in atoms))', 'return ",".join(str(int(i)) for i in (int(x) for x in atoms))', "rule_key_mask"),
    ("D02_dedup_ignores_direction", D, 'out["_key"] = [rule_key(ids) + "|" + str(d) for ids, d in zip(out["atoms"], out["direction"])]', 'out["_key"] = [rule_key(ids) for ids in out["atoms"]]', "rule_key_mask"),
    ("D03_repeated_atom_allowed", D, "keep.append(len(set(ids)) == len(ids) and rule_compatible(objs))", "keep.append(rule_compatible(objs))", "rule_key_mask"),
    ("D04_apriori_no_compat_check", D, "if all(compatible(aset.atoms[aj], x) for x in pobjs):", "if True:", "apriori_survivors"),
    ("D05_apriori_parent_atom_reused", D, "                    if aj in pa:\n                        continue", "                    if False:\n                        continue", "apriori_survivors"),
    ("T01_week_blocks_by_day", T, '(index.normalize() - pd.to_timedelta(index.dayofweek, unit="D")).asi8', "(index.normalize()).asi8", "week_blocks"),
    ("T04_empty_interval_allowed", T, "if le is not None and gt is not None and not (gt < le):", "if le is not None and gt is not None and gt < le:", "canonical_atoms"),
    ("T05_lookup_ignores_operator", T, "i = lookup.get((f, op, float(thr)))", 'i = lookup.get((f, "<=", float(thr)))', "canonical_atoms"),
    ("T06_grow_splits_on_worst_gain", T, "k = int(np.argmax(gain))", "k = int(np.argmin(gain))", "grow_splits"),
    ("T07_grow_min_leaf_halved", T, "if depth >= int(max_depth) or n < 2 * int(min_leaf):", "if depth >= int(max_depth) or n < int(min_leaf):", "grow_splits"),
    ("T08_grow_branches_flipped", T, "left = np.asarray(M[k, idx]).astype(np.float64) > 0.5", "left = np.asarray(M[k, idx]).astype(np.float64) < 0.5", "grow_splits"),
    ("T10_hits_threshold_off_by_one", T, 'if int(rec["hits"]) < min_hits:', "if int(rec[\"hits\"]) < max(1, min_hits - 1):", "tree_engine"),
    ("T11_nonpositive_train_mean_kept", T, "if best_fam is None or best_mean <= 0:", "if best_fam is None or best_mean < 0:", "tree_engine"),
    ("G01_parent_order_wrong", G, "order = [len(ids) - 1, 0] + list(range(1, len(ids) - 1))", "order = list(range(len(ids)))", "lift_over_parents"),
    ("G02_child_drops_first_atom", G, "    child = masks[0].copy()\n    for m in masks[1:]:\n        child &= m", "    child = masks[1].copy()\n    for m in masks[2:]:\n        child &= m", "lift_over_parents"),
    ("G03_min_t_is_max", G, 'out["lift_min_t"] = float(np.min(out["ts"])) if finite else float("nan")', 'out["lift_min_t"] = float(np.max(out["ts"])) if finite else float("nan")', "lift_over_parents"),
    ("G05_lift_stats_ignores_rows", G, "r = lift_stats(parent, child, y, int(hold), rows=rows, block=blk)", "r = lift_stats(parent, child, y, int(hold), rows=slice(None), block=blk)", "lift_over_parents"),
    ("C01_pair_lift_parents_swapped", C, "order = [k - 1, 0] + [i for i in range(1, k - 1)]", "order = [0, k - 1] + [i for i in range(1, k - 1)]", "lift_over_parents"),
    ("P02_stage6_ignores_deep_rules", P, "if up05b is not None and deep_path.exists():", "if False:", "stage5b_pipeline"),
    ("P03_5b_scope_note_constant", P, '"null_scope": deep_scope_note(cfg),', '"null_scope": "none",', "stage5b_pipeline"),
    ("S01_scope_all_or_nothing", D, "if ap_on and tr_on:", "if ap_on or tr_on:", "scope_note"),
]


def run(name, f, old, new, k):
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("output", "data", "__pycache__", "*.egg-info", ".pytest_cache"))
    p = DST / f
    s = p.read_text()
    if s.count(old) != 1:
        return f"BAD-PATTERN({s.count(old)})"
    p.write_text(s.replace(old, new))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", TST, "-x", "-q", "-p", "no:cacheprovider", "-k", k], cwd=DST, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return ("CAUGHT" if r.returncode == 1 else "SURVIVED" if r.returncode == 0 else f"ERROR rc={r.returncode}") + " | " + tail[:90]


if __name__ == "__main__":
    (SRC / "output").mkdir(exist_ok=True)
    res = json.loads(RES.read_text()) if RES.exists() else {}
    pref = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, f, old, new, k in M:
        if not name.startswith(pref) or name in res:
            continue
        res[name] = run(name, f, old, new, k)
        RES.write_text(json.dumps(res, indent=1))
        print(name, res[name], flush=True)
