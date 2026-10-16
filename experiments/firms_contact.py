"""First contact with real data: VIIRS S-NPP active fires, 2024 (docs/design/firms-2024-contact.md).

    python experiments/firms_contact.py

Implements §3-§5 of the protocol exactly. Writes results/firms_2024.json and .md.
"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from run_decoy_benchmark import cp
from scipy.cluster.hierarchy import fcluster, linkage

from argus_autonomy import NOMINAL
from argus_autonomy.config import DecoyConfig
from argus_autonomy.decision import make_decoy_scenario

ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = ("Germany", "Greece", "Portugal")
R_EARTH = 6371.0
PASS_GAP_MIN = 30
CUE_LINK_KM = 0.75
SITE_LINK_KM = 1.0
PERSIST_DAYS, PERSIST_SPAN = 10, 60
RECUR_KM, RECUR_H = 1.0, (6.0, 18.0)
SIM_SEEDS = range(500)
EXPL_RECUR_H = (18.0, 30.0)
# Wide-camera swath 7 cells x 25 m at 10 m/s: the area an ideal lawnmower covers per second.
SWEEP_KM2_PER_S = 7 * 0.025 * 0.010


def load(country: str) -> list[dict]:
    with gzip.open(ROOT / "data" / "firms" / f"viirs-snpp_2024_{country}.csv.gz", "rt") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["t"] = datetime.strptime(r["acq_date"] + r["acq_time"].zfill(4), "%Y-%m-%d%H%M")
        r["lat"], r["lon"] = float(r["latitude"]), float(r["longitude"])
        r["area"] = float(r["scan"]) * float(r["track"])
        r["frp"] = float(r["frp"])
        r["type"] = int(r["type"])
    rows.sort(key=lambda r: r["t"])
    return rows


def xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def single_linkage(lat, lon, km: float) -> np.ndarray:
    """Cluster labels, single linkage at great-circle distance `km` (chord threshold)."""
    if len(lat) == 1:
        return np.array([1])
    chord = 2 * np.sin(km / (2 * R_EARTH))
    return fcluster(linkage(xyz(lat, lon), "single"), chord, criterion="distance")


def haversine_km(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def cues(rows: list[dict]) -> list[dict]:
    passes, cur = [], [rows[0]]
    for r in rows[1:]:
        if (r["t"] - cur[-1]["t"]).total_seconds() > PASS_GAP_MIN * 60:
            passes.append(cur)
            cur = []
        cur.append(r)
    passes.append(cur)
    out = []
    for pid, p in enumerate(passes):
        lab = single_linkage(np.array([r["lat"] for r in p]), np.array([r["lon"] for r in p]), CUE_LINK_KM)
        for k in np.unique(lab):
            ds = [r for r, l in zip(p, lab) if l == k]
            types = {r["type"] for r in ds}
            cls = "static" if types == {2} else "vegetation" if types == {0} else "other"
            out.append({
                "pass": pid, "t": min(r["t"] for r in ds), "n_det": len(ds),
                "lat": float(np.mean([r["lat"] for r in ds])), "lon": float(np.mean([r["lon"] for r in ds])),
                "area_km2": float(sum(r["area"] for r in ds)), "frp": float(sum(r["frp"] for r in ds)),
                "cls": cls, "day": ds[0]["t"].date(),
                "night": sum(r["daynight"] == "N" for r in ds) * 2 > len(ds),
            })
    return out


def add_sites(cs: list[dict]) -> None:
    lab = single_linkage(np.array([c["lat"] for c in cs]), np.array([c["lon"] for c in cs]), SITE_LINK_KM)
    days, span = {}, {}
    for c, s in zip(cs, lab):
        c["site"] = int(s)
        days.setdefault(s, set()).add(c["day"])
    for s, d in days.items():
        span[s] = (max(d) - min(d)).days
    for c in cs:
        c["site_days"] = len(days[c["site"]])
        c["site_span"] = span[c["site"]]
        c["persistent_site"] = c["site_days"] >= PERSIST_DAYS and c["site_span"] >= PERSIST_SPAN


def add_recurrence(cs: list[dict], window=RECUR_H, key="recurs") -> None:
    t = np.array([c["t"].timestamp() for c in cs]) / 3600.0
    lat, lon = np.array([c["lat"] for c in cs]), np.array([c["lon"] for c in cs])
    for i, c in enumerate(cs):
        later = (t >= t[i] + window[0]) & (t <= t[i] + window[1])
        idx = np.flatnonzero(later)
        c[key] = bool(idx.size and (haversine_km(lat[i], lon[i], lat[idx], lon[idx]) <= RECUR_KM).any())


def rate(k, n):
    return {"k": int(k), "n": int(n), "p": k / n if n else float("nan"), "ci95": cp(k, n)}


def pct(r):
    lo, hi = r["ci95"]
    return f"{100 * r['p']:.1f} % ({r['k']}/{r['n']}; {100 * lo:.1f}–{100 * hi:.1f})"


def sim_cues() -> dict:
    cfg, dcfg = NOMINAL, DecoyConfig()
    cell_km2 = (cfg.cell_m / 1000) ** 2
    out = {}
    for fams in (("A", "C"), ("B",)):
        a = []
        for fam in fams:
            for s in SIM_SEEDS:
                sc = make_decoy_scenario(cfg, dcfg, fam, s)
                a.append(float((sc.prior >= cfg.prior.p_hotspot).sum() * cell_km2))
        a = np.array(a)
        out[" ∪ ".join(fams)] = {"n": len(a), "median_km2": float(np.median(a)),
                                 "p_empty": float((a == 0).mean()),
                                 "median_nonempty_km2": float(np.median(a[a > 0])) if (a > 0).any() else None}
    return out


def main():
    sim = sim_cues()
    sim_med = sim["A ∪ C"]["median_km2"]
    res = {"sim": sim, "countries": {}}
    for country in COUNTRIES:
        rows = load(country)
        cs = cues(rows)
        add_sites(cs)
        add_recurrence(cs)
        add_recurrence(cs, EXPL_RECUR_H, "recurs_next_night")   # exploratory, after E13 (see md)
        n = len(cs)
        by = {k: [c for c in cs if c["cls"] == k] for k in ("static", "vegetation", "other")}
        veg, st = by["vegetation"], by["static"]
        va = np.array([c["area_km2"] for c in veg])
        nd = np.array([c["n_det"] for c in veg])
        r = {
            "detections": len(rows), "passes": len({c["pass"] for c in cs}), "cues": n,
            "detections_by_type": {str(k): sum(x["type"] == k for x in rows) for k in (0, 1, 2, 3)},
            "cues_by_class": {k: len(v) for k, v in by.items()},
            "E10_static_share": rate(len(st), n),
            "veg_area_median_km2": float(np.median(va)),
            "veg_area_iqr_km2": [float(np.percentile(va, 25)), float(np.percentile(va, 75))],
            "veg_single_detection_share": rate(int((nd == 1).sum()), len(veg)),
            "veg_ndet_p90": float(np.percentile(nd, 90)),
            "E11_ratio_to_sim": float(np.median(va) / sim_med),
            "E12_veg_on_persistent_site": rate(sum(c["persistent_site"] for c in veg), len(veg)),
            "persistent_veg_sites": len({c["site"] for c in veg if c["persistent_site"]}),
            "veg_sites": len({c["site"] for c in veg}),
            "E13_static_recur": rate(sum(c["recurs"] for c in st), len(st)),
            "E13_veg_recur": rate(sum(c["recurs"] for c in veg), len(veg)),
            "veg_recur_on_persistent": rate(sum(c["recurs"] for c in veg if c["persistent_site"]),
                                            sum(c["persistent_site"] for c in veg)),
            "veg_recur_off_persistent": rate(sum(c["recurs"] for c in veg if not c["persistent_site"]),
                                             sum(not c["persistent_site"] for c in veg)),
        }
        pv = [c for c in veg if c["persistent_site"]]
        ov = [c for c in veg if not c["persistent_site"]]
        r["exploratory"] = {
            "static_night_share": rate(sum(c["night"] for c in st), len(st)),
            "static_recur_18_30h": rate(sum(c["recurs_next_night"] for c in st), len(st)),
            "veg_recur_18_30h": rate(sum(c["recurs_next_night"] for c in veg), len(veg)),
            "veg_persistent_night_share": rate(sum(c["night"] for c in pv), len(pv)),
            "veg_other_night_share": rate(sum(c["night"] for c in ov), len(ov)),
            "sweep_time_s_median": float(np.median(va) / SWEEP_KM2_PER_S),
            "sweep_time_s_p75": float(np.percentile(va, 75) / SWEEP_KM2_PER_S),
        }
        r["verdicts"] = {
            "E10": (r["E10_static_share"]["p"] >= 0.5) if country == "Germany" else (r["E10_static_share"]["p"] <= 0.2),
            "E11": r["E11_ratio_to_sim"] >= 2.0,
            "E12": r["E12_veg_on_persistent_site"]["p"] >= 0.05,
            "E13": (r["E13_static_recur"]["p"] >= 0.5 if st else None, r["E13_veg_recur"]["p"] <= 0.3),
        }
        res["countries"][country] = r

    C = res["countries"]
    L = ["# First contact with real data: VIIRS S-NPP active fires, 2024", "",
         ("Generated by `experiments/firms_contact.py` following `docs/design/firms-2024-contact.md` "
          "(protocol committed before analysis). Rates: percentage (count; Clopper–Pearson 95 % CI)."), "",
         "## Data", "", "| | " + " | ".join(COUNTRIES) + " |", "|---|---|---|---|"]
    for key, lab in (("detections", "detections"), ("passes", "passes"), ("cues", "cues")):
        L.append(f"| {lab} | " + " | ".join(str(C[c][key]) for c in COUNTRIES) + " |")
    for k, lab in (("0", "type 0, vegetation"), ("2", "type 2, static source"), ("1", "type 1, volcano"), ("3", "type 3, offshore")):
        L.append(f"| detections, {lab} | " + " | ".join(str(C[c]["detections_by_type"][k]) for c in COUNTRIES) + " |")
    for k in ("vegetation", "static", "other"):
        L.append(f"| cues, {k} | " + " | ".join(str(C[c]["cues_by_class"][k]) for c in COUNTRIES) + " |")

    def ok(v):
        return "—" if v is None else ("met" if v else "**not met**")

    L += ["", "## E10: share of cues that are labelled static sources", "",
          "Expected: Germany ≥ 50 %; Greece, Portugal ≤ 20 %.", "", "| | share | verdict |", "|---|---|---|"]
    L += [f"| {c} | {pct(C[c]['E10_static_share'])} | {ok(C[c]['verdicts']['E10'])} |" for c in COUNTRIES]
    L += ["", "## E11: vegetation-cue size against the simulation", "",
          (f"Simulated cue (flagged-pixel area at the overpass), families A ∪ C, {sim['A ∪ C']['n']} scenarios: "
           f"median {sim_med:.3f} km², empty in {100 * sim['A ∪ C']['p_empty']:.1f} %. "
           f"Family B: median {sim['B']['median_km2']:.3f} km². Expected: real median ≥ 2 × simulated."), "",
          "| | median area, km² [IQR] | ratio to simulated | single-detection cues | 90th pct detections | verdict |",
          "|---|---|---|---|---|---|"]
    L += [(f"| {c} | {C[c]['veg_area_median_km2']:.3f} [{C[c]['veg_area_iqr_km2'][0]:.3f}–{C[c]['veg_area_iqr_km2'][1]:.3f}] | "
           f"{C[c]['E11_ratio_to_sim']:.1f} | {pct(C[c]['veg_single_detection_share'])} | {C[c]['veg_ndet_p90']:.0f} | "
           f"{ok(C[c]['verdicts']['E11'])} |") for c in COUNTRIES]
    L += ["", "## E12: vegetation cues on persistent sites (≥ 10 days, span ≥ 60 days)", "",
          "Expected: ≥ 5 % in each country.", "", "| | vegetation cues on persistent sites | persistent sites / all vegetation sites | verdict |",
          "|---|---|---|---|"]
    L += [(f"| {c} | {pct(C[c]['E12_veg_on_persistent_site'])} | {C[c]['persistent_veg_sites']} / {C[c]['veg_sites']} | "
           f"{ok(C[c]['verdicts']['E12'])} |") for c in COUNTRIES]
    L += ["", "## E13: recurrence at the next overpass (6–18 h, ≤ 1 km)", "",
          "Expected: static ≥ 50 %, vegetation ≤ 30 %.", "",
          "| | static cues recur | vegetation cues recur | verdict (static, vegetation) | vegetation on persistent sites | vegetation elsewhere |",
          "|---|---|---|---|---|---|"]
    L += [(f"| {c} | {pct(C[c]['E13_static_recur'])} | {pct(C[c]['E13_veg_recur'])} | "
           f"{ok(C[c]['verdicts']['E13'][0])}, {ok(C[c]['verdicts']['E13'][1])} | "
           f"{pct(C[c]['veg_recur_on_persistent'])} | {pct(C[c]['veg_recur_off_persistent'])} |") for c in COUNTRIES]
    L += ["", "## Exploratory (not pre-registered; added after E13 failed)", "",
          ("Static detections are almost all at night, and S-NPP alternates night and day passes, so the "
           "6–18 h window of E13 compares a night pass with the next day pass. The checks below were chosen "
           "after seeing that result; treat them as hypotheses, not confirmations."), "",
          "| | static cues at night | static recur 18–30 h | vegetation recur 18–30 h | night share, vegetation on persistent sites | night share, other vegetation |",
          "|---|---|---|---|---|---|"]
    L += [(f"| {c} | " + " | ".join(pct(C[c]["exploratory"][k]) for k in (
        "static_night_share", "static_recur_18_30h", "veg_recur_18_30h",
        "veg_persistent_night_share", "veg_other_night_share")) + " |") for c in COUNTRIES]
    L += ["", ("Ideal sweep time of a vegetation cue (area / swath rate, 175 m swath at 10 m/s, no turns): "
               + "; ".join(f"{c} median {C[c]['exploratory']['sweep_time_s_median']:.0f} s, 75th pct "
                           f"{C[c]['exploratory']['sweep_time_s_p75']:.0f} s" for c in COUNTRIES)
               + f"; simulated cue (A ∪ C) median {sim_med / SWEEP_KM2_PER_S:.0f} s.")]
    (ROOT / "results" / "firms_2024.json").write_text(json.dumps(res, indent=1, default=str))
    (ROOT / "results" / "firms_2024.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
