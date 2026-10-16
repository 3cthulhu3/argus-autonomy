# Robustness: the v1.1 operating point under VIIRS-sized satellite pixels

Status: **protocol fixed 2026-10-16 (commit `11c313f`), before any mission with the new pixel size was flown. Results in §6.** The only runs before this commit were a shape check of the scenario generator on seeds 5000–5004, which looked at flagged-pixel counts and not at decisions.

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

Full tables: [`results/robustness_viirs_pixel.md`](../../results/robustness_viirs_pixel.md). M3, held-out seeds, paired with 200 m pixels on identical seeds.

| at the operating point (a = 6, b = −10) | 200 m | 400 m |
|---|---|---|
| false alerts on a decoy cue (1000) | 1.2 % (0.6–2.1) | **0.0 %** (0–0.4) |
| real fire CLEARed (1000) | 0.2 % | 0.6 % (0.2–1.3) |
| time to CLEAR a decoy cue, median | 6.00 min | **5.33 min** |
| time to correct alert, median | 2.17 min | 2.33 min |
| undecided (B / A ∪ C) | 2 / 0 | 0 / 0 |
| geofence violations · returns to base | 0 · 2000/2000 | 0 · 2000/2000 |

At the a-priori thresholds the same shift is stronger: false alerts 18 → 1 (exact McNemar p < 10⁻⁴), real fires CLEARed 2 → 13 (p = 0.003).

**Pre-registered expectations:**
- **E14 refuted, in direction.** The time to CLEAR fell rather than rose: −0.67 min in median, faster in 568 and slower in 383 paired missions (sign test p < 10⁻⁸). The reasoning behind E14 was that the CLEAR gate makes dismissal coverage-limited. It is not: in v1.1 the median CLEAR (4.5–6 min) is far longer than the time needed to sweep even a 400 m pixel (about 1.5 min). Dismissal is limited by *evidence*. A larger cue holds more cells that the drone sees cold, and under the independent-cell model each cold cue cell adds evidence against fire. The threshold is reached sooner.
- **E15 met.** No false alert in 1000 decoy missions.
- **E16 not met.** 6 of 1000 real fires CLEARed against 2 (5 vs 1 discordant, p = 0.22 at this n). At ±log 99 the rise is significant.
- **E17 not met.** The median alert time rose by 0.17 min, not 0.25. The paired shift is consistent in direction (slower in 441, faster in 115) but small: the median paired difference is zero.

**Why fires are missed (post hoc, not pre-registered).** The 13 missions that CLEAR a fire at either threshold pair are 7 distinct fires, because families A and C share a fire on the same seed. In all 7, Λ never rose above its prior value of 0 before the CLEAR. In 6, at most a third of the burning area at launch lay inside the flagged pixels (none of it in 3). In 12 of the 13 missions, Λ climbed to between 124 and 1613 nats later in the sweep run; in the 13th (A 684) the fire was never found. These are the coverage misses of v1.1 (§14 of the v1.1 design doc), made more frequent: the larger cue yields cold evidence faster, so the drone dismisses the cue before it reaches the ring where the fire is.

**What this means.**
1. **The v1.1 operating point does not transfer across pixel sizes.** It moves along the trade-off between false alerts and missed fires: with 400 m pixels, false alerts vanish and missed fires rise (2 → 6 at the operating point, not significant at this n; 2 → 13 at ±log 99, significant). Thresholds chosen in one satellite geometry should not be deployed in another without re-selection.
2. **The FIRMS note's inference about time to CLEAR was wrong** ([`firms-2024-contact.md`](firms-2024-contact.md), §9, point 3). Real cues being larger does not make the simulated CLEAR optimistic. It makes the simulated *missed-fire rate* optimistic. That note is amended accordingly.
3. **The design question for v1.2 is sharper.** Coverage misses are now the dominant error in every configuration tested. A CLEAR that is a statement about the flagged pixels only becomes riskier as the pixels grow. Candidates are a gate over the ring as well, which was too slow in v1.1, §12, or a short search of the ring after the test would clear. Each would be pre-registered and run on selection seeds first.
