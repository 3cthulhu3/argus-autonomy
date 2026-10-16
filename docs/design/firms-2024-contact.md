# First contact with real data: VIIRS active fires, 2024

Status: **protocol fixed 2026-10-16 (commit `9129bf9`), before any distribution in the data was computed or looked at;** at that point only the column headers and row counts had been read. **Results in §9.**

## 1. Why

Every number in v1.0 and v1.1 is about the method's behaviour in a world whose satellite cues, decoys and persistence are *assumed*. This note asks which of those assumptions the public satellite record can check, checks them, and states which it cannot.

## 2. Data

NASA FIRMS archive, VIIRS S-NPP 375 m active-fire detections, calendar year 2024, country files for Germany, Greece and Portugal ([`data/firms/`](../../data/firms/), gzipped, unmodified). Columns used: `latitude`, `longitude`, `scan`, `track` (pixel size, km), `acq_date`, `acq_time` (UTC), `frp` (MW), `daynight`, `type`.

`type` is FIRMS's own label: 0 presumed vegetation fire, 1 active volcano, 2 other static land source (industry, gas flares), 3 offshore. It is assigned after the fact from detection persistence, so it is **not** available to a dispatcher at the moment of a cue. Here it serves as an approximate ground truth for one class of decoy.

## 3. Units of analysis

- **Pass.** Detections of one country sorted by time; a gap of more than 30 min starts a new pass. One pass is one overpass of S-NPP.
- **Cue.** Within a pass, detections linked by single linkage at 750 m (two VIIRS pixels), haversine distance. A cue is what one dispatch would start from. Area = Σ scan × track over its detections.
- **Cue class.** *static* if every detection has type 2; *vegetation* if every detection has type 0; *other* otherwise (mixed, volcano, offshore), counted but not interpreted.
- **Site.** Cue centroids over the whole year linked by single linkage at 1 km. Per site: number of distinct days with a cue, and span (last day − first day).
- **Persistent vegetation site.** A site containing a vegetation cue, seen on ≥ 10 distinct days with a span of ≥ 60 days. A wildfire does not burn in one place for months; such a site is most likely an unflagged static source or a field burnt repeatedly.
- **Recurrence.** A cue recurs if some cue in a *later* pass of the same country lies within 1 km of its centroid and between 6 and 18 h after it (the next S-NPP overpass).

## 4. Simulation side

The same quantities from the scenario generator, without flying any mission: flagged-pixel area at the overpass (cells with prior ≥ `p_hotspot`, × 625 m²), families A ∪ C and B, seeds 0–499 each.

## 5. Questions and pre-registered expectations

| | question | simulation assumption it checks | expectation |
|---|---|---|---|
| E10 | What share of cues are labelled static sources? | Family B exists: the cue is sometimes a non-spreading source | Germany ≥ 50 %; Greece and Portugal each ≤ 20 % |
| E11 | How large is a real vegetation cue? | Cue size: flagged 200 m pixels around a fire 9 min old | In each country, the median vegetation-cue area is ≥ 2 × the median simulated cue area (A ∪ C) |
| E12 | Do persistent sources hide among unlabelled detections? | Decoys still reach the drone after known static sources are masked | In each country, ≥ 5 % of vegetation cues lie on persistent vegetation sites |
| E13 | Is persistence the decoy's signature at satellite cadence? | §1 of the v1.1 design: decoys persist, fires spread and die | Static cues recur at the next overpass in ≥ 50 % of cases; vegetation cues in ≤ 30 %, in each country |

Each rate is reported with a Clopper–Pearson 95 % interval; an expectation counts as met only by its point estimate, and the interval is shown alongside.

## 6. What this data cannot check

- **Anything inside a 30-minute mission.** S-NPP sees a place about twice a day. Within-mission growth, the evidence the v1.1 test runs on, is invisible at that cadence.
- **Agricultural and other contained burns** (the hot decoy H). They are labelled type 0 like wildfires. Only E12 and E13 see them, indirectly.
- **Warm decoys** (sunlit rock, vehicles). The VIIRS active-fire algorithm is built to reject them; they are a problem of the drone's camera, not of the satellite.
- **Latency and geolocation error.** The archive has no ground truth for either.

## 7. Use of the results

Analysis only. No simulation parameter is changed in this note. If a parameter should change, that is proposed separately and re-run under the v1.1 protocol.

## 8. Source

NASA FIRMS (Fire Information for Resource Management System), VIIRS S-NPP 375 m active-fire archive, accessed October 2026. Algorithm: Schroeder et al., *Remote Sensing of Environment* 143, 2014.

## 9. Results

Full tables: [`results/firms_2024.md`](../../results/firms_2024.md). No expectation held in all three countries.

| | Germany | Greece | Portugal |
|---|---|---|---|
| cues (dispatch-sized clusters) | 7615 | 2250 | 2215 |
| **E10** labelled static sources, share of cues | 70.9 % ✓ | 23.6 % ✗ (≤ 20 % expected) | 8.2 % ✓ |
| **E11** median vegetation-cue area / simulated | 1.9 × ✗ (≥ 2 expected) | 2.8 × ✓ | 6.5 × ✓ |
| single-detection vegetation cues | 80.6 % | 67.9 % | 45.1 % |
| **E12** vegetation cues on persistent sites | 15.1 % ✓ | 5.8 % ✓ | 1.2 % ✗ |
| **E13** static cues recur at the next overpass | 32.2 % ✗ | 0.2 % ✗ | 1.1 % ✗ |
| **E13** vegetation cues recur at the next overpass | 1.8 % ✓ | 6.5 % ✓ | 23.3 % ✓ |

**E13 failed because the window was wrong, not because persistence is absent.** Labelled static sources are detected almost only at night (79 % of cues in Germany, 99 % in Greece and Portugal), and S-NPP alternates night and day passes, so a 6–18 h window pairs a night detection with a day pass that cannot see it. The code was checked; the error is in the protocol. A window of 18–30 h (the next night), chosen *after* seeing this and therefore exploratory, gives static recurrence of 46 %, 39 % and 29 % against 5–8 % for vegetation cues. Persistence does separate the two classes at satellite cadence, by a factor of 4 to 9, but a static source is seen on only a third to a half of nights. It is a statistical signature, not a deterministic one, which is how the v1.1 sensor model already treats it (hit rate 0.70–0.85 per look, not 1).

**What the results say about the simulation.**

1. **Most real decoy cues are already labelled, and therefore maskable.** In Germany 71 % of all cues are known static sources. A dispatcher would remove them with a lookup table before any drone flies; the drone's discrimination problem begins with what the mask misses. The v1.1 family B is that residual, not the raw stream.
2. **The residual is not empty.** After masking, 15 % (Germany), 6 % (Greece) and 1 % (Portugal) of the remaining cues sit on sites seen on 10 or more days over at least two months. Exploratory: these sites are night-dominated (92 %, 75 %, 100 %) like the labelled static sources and unlike other vegetation cues (25 %, 19 %, 47 %), so they are most likely unflagged industrial sources. These shares are **lower bounds** on the decoy share: agricultural and other contained burns, the hot decoys of v1.1, carry the same label as wildfires and are invisible to this analysis. The data therefore neither support nor contradict the test's dispatch prior of 0.5; they bound the decoy share from below only.
3. **Real cues are 2–6.5 times larger than simulated ones, and this is the finding with consequences.** The simulation flags 200 m pixels around a fire nine minutes old; VIIRS pixels are 375 m, and between 45 % and 81 % of real vegetation cues are a single pixel, so the difference is largely resolution. It matters because the CLEAR gate requires every flagged pixel to be seen. Exploratory: an ideal sweep of the median real cue takes 87 s (Germany), 130 s (Greece) and 297 s (Portugal), against 46 s for the simulated one; the 75th percentile in Portugal takes 11 minutes, more than a third of the endurance. The v1.1 time to CLEAR (6 min at the operating point) is therefore optimistic for real cues, and the gap grows with fire activity.

**Update, same day: point 3 was tested and its inference about time to CLEAR was wrong.** With 400 m pixels the time to CLEAR fell, because dismissal in v1.1 is limited by evidence, not by coverage. What larger cues raise is the rate of missed fires, through faster dismissal before the ring around the cue is searched ([`robustness-viirs-pixel.md`](robustness-viirs-pixel.md)). The sentence above is kept as written, because it is what was predicted.

**Proposed, then done.** Re-run of the v1.1 held-out benchmark with VIIRS-sized pixels, pre-registered separately: [`robustness-viirs-pixel.md`](robustness-viirs-pixel.md).
