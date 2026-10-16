# First contact with real data: VIIRS active fires, 2024

Status: **protocol fixed 2026-10-16, before any distribution in the data was computed or looked at.** At the time of writing, only the column headers and the row counts of the three files had been read.

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
