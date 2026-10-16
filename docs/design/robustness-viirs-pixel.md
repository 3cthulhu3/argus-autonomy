# Robustness: the v1.1 operating point under VIIRS-sized satellite pixels

Status: **protocol fixed 2026-10-16, before any mission with the new pixel size was flown.** The only runs before this commit were a shape check of the scenario generator on seeds 5000–5004, which looked at flagged-pixel counts and not at decisions.

## 1. Why

The FIRMS analysis ([`firms-2024-contact.md`](firms-2024-contact.md), §9) found real vegetation cues 2–6.5 times larger than simulated ones, largely because VIIRS pixels are 375 m and the simulation's are 200 m. The CLEAR gate forbids dismissing a cue before all of it has been seen, so the simulated time to CLEAR may be optimistic. This note measures that, and checks whether the v1.1 operating point still holds.

## 2. Change

One change to the satellite model, nothing else:

| | v1.1 | this note |
|---|---|---|
| `coarse_factor` | 8 cells (200 m) | **16 cells (400 m)** |
| `detect_fraction` | 0.05 | **0.0125** |

- **Why 400 m and not 375 m.** 375 m is 15 cells, which does not divide the 64-cell grid. 400 m does, and it is closer to the size of a real detection: the median single-detection cue in the FIRMS data covers about 0.15 km², a 390 m square, because VIIRS pixels grow away from nadir.
- **Why 0.0125.** It keeps the detection threshold at the same *absolute* burning area (0.05 × 64 = 0.0125 × 256 = 3.2 cells). Only the localisation granularity changes, not the sensitivity.

Estimator, test, planner, thresholds and seeds are unchanged. The estimator's priors are built from the new map exactly as before (`p_hotspot` in flagged pixels, `p_ring` in the ring).

## 3. Runs

M3 with the CLEAR gate, `steer_clear` = −log 99, held-out seeds (B 1000–1999, A and C 500–999), flown on-line at two threshold pairs:
- ±log 99 (a priori);
- a = 6, b = −10 (the v1.1 operating point).

**Baseline:** the same missions with 200 m pixels, taken as first-passage decisions from the v1.1 held-out sweep log (`results/sweep_heldout.jsonl.gz`). That log agrees with on-line flights in 2000 of 2000 missions.

**Paired tests on identical seeds:**
- decisions: exact McNemar on discordant missions;
- times: two-sided sign test over missions decided in both conditions with different times.

## 4. Pre-registered expectations

At the operating point, 400 m pixels against 200 m:
- **E14.** Median time to CLEAR a decoy cue (family B) rises by at least 1.0 min.
- **E15.** False alerts stay at or below 2 % (point estimate).
- **E16.** Real fires dismissed (A ∪ C) stay at or below 2 of 1000. The gate now covers more ground, so coverage misses should not rise.
- **E17.** Median time to correct alert (A ∪ C) rises by at least 0.25 min, because the prior spreads the same mass over four times the area.

The undecided rate is also reported. If it exceeds 5 % in either B or A ∪ C, the operating point fails the v1.1 selection rule under 400 m pixels, whatever E14–E17 say.

## 5. Known limitation of the test itself

The grid is 1.6 km, so 400 m pixels make it only 4 × 4 pixels. The ring of one pixel around a flagged pixel (prior `p_ring`) then covers up to half of the grid, and flagged regions near an edge are clipped. A larger world would be the cleaner test. This one measures the direction and rough size of the effect, not its value in the field.

## 6. Results

*(to be filled after the run)*
