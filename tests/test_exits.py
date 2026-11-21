"""M6: exit grid, smoothed surface, plateau choice, baseline comparison, F3, stage 08."""

import copy
import re

import numpy as np
import pandas as pd
import pytest

from edge_research.config import load_config
from edge_research.discovery.candidates import build_candidates
from edge_research.discovery.exits import build_exits, exit_grid, select_exits
from edge_research.discovery.filters import run_filters
from edge_research.discovery.pairs import coarse_screen
from edge_research.labels.builder import family_list
from edge_research.synth.generator import make_synthetic_bars
from test_discovery import CFG, DEEP_OFF, STRONG, _world

NAMES = [n for n, _ in family_list(CFG)]


def _fam(means, n_ep=400, t=3.0, cand_id=0, direction="long", min_over=None):
    rows = []
    for n in NAMES:
        rows.append({"cand_id": cand_id, "direction": direction, "family": n, "mean": means.get(n, 0.0) if isinstance(means, dict) else means,
                     "n_episodes": n_ep if min_over is None or n not in min_over else min_over[n], "t": t})
    return pd.DataFrame(rows)


def _cfg(**over):
    c = copy.deepcopy(CFG)
    c["exits"].update(over)
    return c


# ---------------------------------------------------------------- grid
def test_exit_grid_neighbourhoods_by_hand():
    g = exit_grid(NAMES)
    assert len(g) == 42                                                           # 6 T + 18 B + 18 I (intrabar_families: true)
    assert g["T1"]["smooth"] == ["T1", "T2"] and g["T3"]["smooth"] == ["T2", "T3", "T6"] and g["T24"]["smooth"] == ["T12", "T24"]
    assert g["T3"]["f3"] == ["T1", "T2", "T3", "T6", "T12"] and g["T1"]["f3"] == ["T1", "T2", "T3"]
    assert sorted(g["B2_1p5_12"]["smooth"]) == sorted(f"B{tp}_{sl}_12" for tp in ("1", "2", "3") for sl in ("1", "1p5", "2"))          # centre: 3 x 3
    assert sorted(g["B1_1_12"]["smooth"]) == sorted(["B1_1_12", "B1_1p5_12", "B2_1_12", "B2_1p5_12"])                                 # corner: 2 x 2
    assert sorted(g["B2_1p5_12"]["f3"]) == sorted(["B2_1p5_12", "B1_1p5_12", "B3_1p5_12", "B2_1_12", "B2_2_12"])                       # self + 4
    assert sorted(g["B1_1_12"]["f3"]) == sorted(["B1_1_12", "B2_1_12", "B1_1p5_12"])
    assert all(n.startswith("B") for n in g["B2_1p5_12"]["smooth"] + g["B2_1p5_12"]["f3"])                                             # B and I are separate grids...
    assert sorted(g["I2_1p5_12"]["smooth"]) == sorted(f"I{tp}_{sl}_12" for tp in ("1", "2", "3") for sl in ("1", "1p5", "2"))          # ...with the same shape
    assert all(n.endswith("_24") for n in g["B3_2_24"]["smooth"]) and g["B3_2_24"]["hold"] == 24 and g["T6"]["hold"] == 6      # horizons are never mixed


def _coords(n):
    m = re.match(r"^[BI]([0-9p]+)_([0-9p]+)_(\d+)$", n)
    return tuple(float(x.replace("p", ".")) for x in m.groups()) if m else None


def _ref_smooth(means, n):
    """Independent smoothing straight from the coordinates."""
    c = _coords(n)
    if c is None:
        k = int(n[1:])
        ns = sorted(int(x[1:]) for x in means if x.startswith("T"))
        i = ns.index(k)
        return np.mean([means[f"T{ns[j]}"] for j in (i - 1, i, i + 1) if 0 <= j < len(ns)])
    tps = sorted({_coords(x)[0] for x in means if _coords(x) and x[0] == n[0]})
    sls = sorted({_coords(x)[1] for x in means if _coords(x) and x[0] == n[0]})
    it, js = tps.index(c[0]), sls.index(c[1])
    vals = []
    for x in means:
        d = _coords(x)
        if d and x[0] == n[0] and d[2] == c[2] and abs(tps.index(d[0]) - it) <= 1 and abs(sls.index(d[1]) - js) <= 1:
            vals.append(means[x])
    return np.mean(vals)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_select_exits_equals_independent_reference(seed):
    rng = np.random.default_rng(seed)
    cfg = CFG
    parts = []
    refs = {}
    for cid in range(60):
        means = {n: float(rng.normal(0.1, 0.3)) for n in NAMES}
        n_ep = {n: int(rng.integers(100, 600)) for n in NAMES}
        refs[cid] = (means, n_ep)
        f = _fam(means, cand_id=cid)
        f["n_episodes"] = [n_ep[n] for n in f["family"]]
        parts.append(f)
    surface, chosen = select_exits(pd.concat(parts), cfg)
    assert len(surface) == 60 * len(NAMES)
    for r in chosen.itertuples():
        means, n_ep = refs[r.cand_id]
        sm = {n: _ref_smooth(means, n) for n in NAMES}
        el = [n for n in NAMES if n_ep[n] >= cfg["candidates"]["min_episodes_train"]]
        if not el:
            assert r.chosen_family is None
            continue
        best = max(el, key=lambda n: (round(sm[n], 12), n.startswith("T"), -int(re.findall(r"(\d+)$", n)[0])))
        tbest = max([n for n in el if n.startswith("T")], key=lambda n: (round(sm[n], 12), -int(n[1:]))) if any(n.startswith("T") for n in el) else None
        exp = best
        if best[0] in "BI" and tbest and sm[best] - sm[tbest] < cfg["exits"]["min_improvement_R"]:
            exp = tbest
        assert r.chosen_family == exp, (r.cand_id, r.chosen_family, exp)
        assert r.chosen_smooth == pytest.approx(sm[exp]) and r.chosen_mean == pytest.approx(means[exp])
        assert r.raw_best_family == max(el, key=lambda n: means[n])


# ---------------------------------------------------------------- the point of smoothing, baseline, F3, eligibility
def test_a_lone_spike_loses_to_a_broad_plateau():
    means = {n: -0.3 for n in NAMES}
    means["B2_1p5_24"] = 2.0                                          # isolated peak: raw best
    means.update({"T6": 0.5, "T12": 0.5, "T3": 0.45, "T24": 0.45})   # broad plateau on the time stops
    surface, chosen = select_exits(_fam(means), CFG)
    r = chosen.iloc[0]
    assert r["raw_best_family"] == "B2_1p5_24" and r["chosen_family"] in ("T6", "T12") and not r["raw_best_is_chosen"]
    assert r["chosen_smooth"] > surface.set_index("family").loc["B2_1p5_24", "smooth"]


def test_barrier_must_beat_the_best_time_stop_by_the_margin():
    base = {n: 0.0 for n in NAMES}
    base.update({"T6": 0.50, "T12": 0.50, "T3": 0.50})
    plateau = lambda v: {f"B{tp}_{sl}_12": v for tp in ("1", "2", "3") for sl in ("1", "1p5", "2")}
    small = {**base, **plateau(0.52)}
    big = {**base, **plateau(0.70)}
    r_small = select_exits(_fam(small), CFG)[1].iloc[0]
    r_big = select_exits(_fam(big), CFG)[1].iloc[0]
    assert r_small["kind"] == "T" and "min_improvement" in r_small["reason"]               # 0.02 < 0.05 -> simpler wins
    assert r_big["kind"] == "B" and r_big["chosen_family"] == "B2_1p5_12" and r_big["improvement_over_T"] >= 0.05
    r_margin = select_exits(_fam(small), _cfg(min_improvement_R=0.01))[1].iloc[0]
    assert r_margin["kind"] == "B"


def test_f3_needs_enough_positive_neighbouring_exits_and_enough_episodes():
    one_cell = {n: -0.2 for n in NAMES}
    one_cell["T6"] = 0.6
    r = select_exits(_fam(one_cell), CFG)[1].iloc[0]
    assert r["chosen_family"] == "T3" and r["f3_cells"] == 5 and r["f3_positive"] == 1 and not r["pass_f3"]      # T3, T6, T12 tie at smooth 0.067: interior, shorter hold
    wide = {n: -0.2 for n in NAMES}
    wide.update({"T3": 0.3, "T6": 0.6, "T12": 0.4})
    r2 = select_exits(_fam(wide), CFG)[1].iloc[0]
    assert r2["chosen_family"] == "T6" and r2["f3_positive"] == 3 and r2["pass_f3"]
    # a neighbour with too few episodes does not count as positive
    r3 = select_exits(_fam(wide, min_over={"T12": 50}), CFG)[1].iloc[0]
    assert r3["f3_positive"] == 2 and not r3["pass_f3"]
    assert not select_exits(_fam(wide), _cfg(f3_min_positive=4))[1].iloc[0]["pass_f3"]


def test_ineligible_cells_cannot_be_chosen_and_all_ineligible_gives_no_exit():
    means = {n: 0.1 for n in NAMES}
    means["T6"] = 5.0
    r = select_exits(_fam(means, min_over={"T6": 10, "T3": 10, "T12": 10}), CFG)[1].iloc[0]
    assert r["chosen_family"] not in ("T6", "T3", "T12") and r["raw_best_family"] != "T6"      # too few episodes -> out of the running
    none = select_exits(_fam(means, n_ep=5), CFG)[1].iloc[0]
    assert none["chosen_family"] is None and not none["pass_f3"] and "no eligible" in none["reason"]
    nan = {n: np.nan for n in NAMES}
    assert select_exits(_fam(nan), CFG)[1].iloc[0]["chosen_family"] is None


def test_tie_between_a_time_stop_and_a_barrier_prefers_the_time_stop():
    means = {n: 0.2 for n in NAMES}
    r = select_exits(_fam(means), _cfg(min_improvement_R=-1.0))[1].iloc[0]
    assert r["kind"] == "T" and r["chosen_family"] == "T2"            # all equal: time stop, interior of the plateau (3-point neighbourhood), shortest hold
    flat_b = {f"B{tp}_{sl}_12": 0.7 for tp in ("1", "2", "3") for sl in ("1", "1p5", "2")}
    assert select_exits(_fam({**{n: 0.0 for n in NAMES}, **flat_b}), CFG)[1].iloc[0]["chosen_family"] == "B2_1p5_12"       # flat barrier plateau: its centre


# ---------------------------------------------------------------- build_exits (representatives, require_f3)
def _passed(rows):
    # since M11 every candidate carries the unified N-atom columns (stage 6 normalises pairs and deep rules to one shape)
    base = {"atoms": [1, 2], "n_atoms": 2, "rule_key": "1,2", "rule_expr": "a AND b", "source": "pairs",
            "atom_a": 1, "atom_b": 2, "expr_a": "a", "expr_b": "b", "direction": "long", "rank_score": 1.0, "mono_score": 0.9, "f_cluster_size": 2,
            "f8_simplified": False, "f8_dropped": 0, "f8_val_mean": np.nan,
            "best_family": "T6", "best_t": 3.0, "best_mean": 0.5}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_representative_is_repicked_among_members_that_pass_f3():
    good = {n: -0.2 for n in NAMES}
    good.update({"T3": 0.3, "T6": 0.6, "T12": 0.4})
    bad = {n: -0.2 for n in NAMES}
    bad["T6"] = 0.9                                                  # best plateau but an isolated exit
    fam = pd.concat([_fam(bad, cand_id=10), _fam(good, cand_id=11), _fam(good, cand_id=12)])
    passed = _passed([{"cand_id": 10, "fcluster": 0, "plateau_score": 0.9}, {"cand_id": 11, "fcluster": 0, "plateau_score": 0.5},
                      {"cand_id": 12, "fcluster": 1, "plateau_score": 0.4}])
    surface, chosen, info = build_exits(passed, fam, CFG)
    assert chosen.set_index("cand_id")["pass_f3"].to_dict() == {10: False, 11: True, 12: True}
    assert chosen.set_index("cand_id")["is_representative"].to_dict() == {10: False, 11: True, 12: True}
    assert info["n_in"] == 3 and info["n_pass_f3"] == 2 and info["n_frozen"] == 2 and info["n_with_exit"] == 3
    _, ch2, i2 = build_exits(passed, fam, _cfg(require_f3=False))
    assert ch2.set_index("cand_id")["is_representative"].to_dict() == {10: True, 11: False, 12: True} and i2["n_frozen"] == 2        # F3 not required: best plateau
    e1, e2, i3 = build_exits(passed.iloc[0:0], fam, CFG)
    assert e1.empty and e2.empty and i3["n_frozen"] == 0


# ---------------------------------------------------------------- planted-edge world and stage 08
@pytest.fixture(scope="module")
def strong_e():
    aset, lab, tr = _world(1, 20000, True)
    surv, _ = coarse_screen(aset, lab, CFG, tr)
    cand, fam, _ = build_candidates(surv, aset, lab, CFG, tr)
    _, passed, _ = run_filters(cand, fam, aset, lab, CFG, tr)
    surface, chosen, info = build_exits(passed, fam, CFG)
    return passed, fam, surface, chosen, info


def test_planted_world_exits(strong_e):
    passed, fam, surface, chosen, info = strong_e
    assert info["n_in"] == len(passed) == len(chosen) and len(surface) == len(NAMES) * len(passed)
    assert info["n_frozen"] > 0 and info["n_pass_f3"] >= info["n_frozen"]
    fz = chosen[chosen["frozen_ok"]]
    assert fz["pass_f3"].all() and (fz["chosen_n_episodes"] >= CFG["candidates"]["min_episodes_train"]).all() and (fz["chosen_mean"] > 0).all()
    assert (fz["f3_positive"] >= 3).all()
    all_max = surface.groupby("cand_id")["mean"].max()
    assert (fz["chosen_smooth"].to_numpy() <= all_max.loc[fz["cand_id"]].to_numpy() + 1e-9).all()               # an average never exceeds the largest cell it averages
    assert chosen.loc[chosen["is_representative"], "fcluster"].is_unique and set(chosen.loc[chosen["is_representative"], "fcluster"]) <= set(fz["fcluster"])
    s = surface.set_index(["cand_id", "family"])
    r = fz.iloc[0]
    assert s.loc[(r.cand_id, r.chosen_family), "smooth"] == pytest.approx(r.chosen_smooth)
    assert surface["eligible"].mean() > 0.3


def test_pipeline_stage8_end_to_end_resume_and_invalidation(tmp_path):
    from edge_research.pipeline import run_stage_8, run_stages_0_8
    from edge_research.synth.generator import write_exporter_csv
    df, meta = make_synthetic_bars(12000, seed=4, plant_edge=STRONG)
    csv = write_exporter_csv(df, tmp_path / "s.csv")
    sp = meta["spread_points"].to_frame()
    sp.insert(0, "time", sp.index.strftime("%Y%m%d %H:%M"))
    sp.to_csv(tmp_path / "s_spread.csv", index=False)
    run = tmp_path / "run"
    sps = tmp_path / "s_spread.csv"
    out = run_stages_0_8(csv, run, CFG, "SYNTH", sps)
    assert out["status"]["08_exits"] == "ran" and out["n_frozen"] > 0
    d8 = run / "08_exits"
    for f in ("exit_surface.csv", "chosen_exits.parquet", "chosen_exits.csv", "exits_info.json", "summary.md"):
        assert (d8 / f).exists(), f
    ch = pd.read_parquet(d8 / "chosen_exits.parquet")
    assert int(ch["is_representative"].sum()) == out["n_frozen"]
    assert set(run_stages_0_8(csv, run, CFG, "SYNTH", sps)["status"].values()) == {"skipped"}
    o2 = run_stages_0_8(csv, run, load_config(overrides={"exits": {"min_improvement_R": 0.5}, **DEEP_OFF}), "SYNTH", sps)
    assert {k for k, v in o2["status"].items() if v == "ran"} == {"08_exits"}
    o3 = run_stages_0_8(csv, run, load_config(overrides={"exits": {"min_improvement_R": 0.5}, "filters": {"plateau_min_ratio": 0.9}, **DEEP_OFF}), "SYNTH", sps)
    assert {k for k, v in o3["status"].items() if v == "ran"} == {"07_filters", "08_exits"}


def test_episode_eligibility_boundary_is_inclusive():
    means = {n: 0.1 for n in NAMES}
    means["T6"] = 5.0
    means["T3"] = means["T12"] = 0.6                     # T6 has the strictly best smoothed value (T3 and T12 would tie with it if the neighbours were equal)
    at = select_exits(_fam(means, min_over={"T6": 200}), CFG)[1].iloc[0]
    below = select_exits(_fam(means, min_over={"T6": 199}), CFG)[1].iloc[0]
    assert at["chosen_family"] == "T6" and below["chosen_family"] != "T6"
