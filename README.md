# argus-autonomy

**Active wildfire-front estimation for a single UAV, in simulation.**
A coarse, stale satellite hotspot gives a prior; a nadir thermal camera gives noisy local measurements; a Bayesian occupancy filter fuses both; an information-rate planner decides where to fly next. A geofence and a return-to-base monitor have the final word on every move.

<p align="center"><img src="media/demo.gif" width="720" alt="Animated mission: hidden ground-truth fire on the left, onboard belief map with drone trajectory on the right"></p>

<sub>Seed 47 is shown because its F1 is the median of the 100 nominal runs. It was picked by `experiments/make_media.py`, not by hand.</sub>

> **Question.** Given a coarse satellite hotspot prior and a noisy onboard thermal sensor, how quickly can a single UAV confirm a wildfire and localise its perimeter with a measurable uncertainty, while staying inside a geofence and keeping enough endurance to return?

This is the autonomy layer of the Argus Aegis wildfire-drone concept. It is a research prototype, **not flight software**: everything runs in a 2-D grid simulation.

## Result in one table

The table shows the nominal condition, 100 Monte Carlo seeds per planner. All planners face *identical* fires on the same seed.

| planner | first confirmation, median | 50 % of burnt area confirmed, median | F1 of map at landing | perimeter error, median |
|---|---|---|---|---|
| lawnmower (full map) | 9.4 min | 15.4 min | 0.829 | 66 m |
| prior-lawnmower (sweeps the satellite hotspot box) | 3.3 min | 6.2 min | 0.878 | 35 m |
| **info-rate (this work)** | **1.8 min** | **3.0 min** | **0.925** | **18 m** |

Against the strongest baseline (prior-lawnmower), paired on identical seeds:
- info-rate confirms the fire earlier in **99/100** seeds (median −1.5 min, Wilcoxon *p* < 10⁻¹⁶);
- its perimeter error is lower in **91/100** seeds.

Cells are 25 m, so a median perimeter error of 18 m is below one cell. Across all 900 benchmark missions there were **0 geofence violations** and **900/900 returns to base**, each with ≥ 7 ticks (70 s) of reserve.

Full table, all three conditions: [`results/summary.md`](results/summary.md).

## Where it breaks (read this part)

Two failure modes are measured, not hidden.

**1. An idealised spread model.** In the `wind-mismatch` condition the estimator assumes calm air while the real fire is wind-driven. Info-rate degrades but still leads:

| | F1 | perimeter error |
|---|---|---|
| info-rate | 0.925 → 0.872 | 18 m → 29 m |
| prior-lawnmower | 0.878 → 0.772 | 35 m → 60 m |

The prediction step alone tells the underlying story. With no observations, the calm-air model recovers only 35 % of the true burnt area after 30 min (IoU 0.28); with wind known it recovers 66 % ([`results/prediction_calibration.md`](results/prediction_calibration.md)). Observations are what make the map good; the model only buys time between them.

**2. Over-trusting the satellite prior.** In `degraded-prior` (older overpass, 6-cell geolocation error), 40/100 overpasses detect nothing. In 42/100 seeds none of the fire lies inside the flagged pixels or the ring around them; 40 of those are the non-detections. A greedy information planner then sees almost no expected gain where the fire actually is:
- info-rate's mean F1 is still the best (0.847);
- but it has **13/100 catastrophic runs** (F1 < 0.7), against 2/100 for the prior-blind lawnmower;
- its F1 is more than 0.02 below the blind lawnmower's in 25/100 seeds.

Tempering the prior fixes most of this. Raising the background prior from 0.002 to 0.01 cuts catastrophic runs from 13 to 3 and costs nothing in nominal conditions (F1 0.925 → 0.929). The fix works, but it is a hand-tuned knob, not a principled treatment of prior misspecification ([`results/prior_tempering.md`](results/prior_tempering.md)).

## In progress: decoys (v1.1)

The estimator above cannot tell a fire from a sunlit rock or a contained burn: to it, a hot spot that stays hot is a fire. A probe with three static decoys confirms at least one as fire in **50/100** missions ([`results/probe_decoys.json`](results/probe_decoys.json)).

v1.1 replaces the per-cell threshold with a sequential test between two models, *a fire that spreads* and *things that only stay hot*, so that growth, and the absence of predicted growth, become the evidence. First results (milestone (i), thresholds fixed a priori, not tuned):

| | v1.0 | two-model test |
|---|---|---|
| false alert when the satellite cue is a decoy (1000 missions) | 100 % | 2.2 % |
| alert placed on a decoy instead of the fire (500) | 9.8 % | 0.0 % |
| real fire dismissed (fire + decoys, 500) | 0 % | **2.8 %** |
| map F1, fire only (500) | 0.926 | 0.926 |

The last error is the one that matters, and it is analysed rather than hidden: in 9 of the 14 misses the drone rejected a decoy in the cue and went home without ever seeing the fire ([`results/decoys_m1_misses.md`](results/decoys_m1_misses.md)). Design, decisions and the full account: [`docs/design/v1.1-decoy-discrimination.md`](docs/design/v1.1-decoy-discrimination.md).

## Method

| component | what it does | file |
|---|---|---|
| Fire | Stochastic cellular automaton with wind (exp(k·cos θ) anisotropy), correlated fuel field and burn-out | [`world.py`](src/argus_autonomy/world.py) |
| Satellite prior | 8×8-cell pixels (200 m), detection threshold, latency, geolocation error | [`perception.py`](src/argus_autonomy/perception.py) |
| Sensor | 7×7-cell footprint, P(detect) = 0.85, P(false alarm) = 0.03, independent cells | [`perception.py`](src/argus_autonomy/perception.py) |
| Filter | Exact log-odds Bayes update. Gated mean-field prediction through the estimator's own spread model: only cells with *b* ≥ 0.5 spread, because ungated mean-field over-predicts the burnt area 4.7× after 30 min | [`perception.py`](src/argus_autonomy/perception.py) |
| Planner | Expected information gain *G(c)* = Σ over the footprint of the mutual information I(X;Z); flies to argmax *G(c)* / (travel ticks + τ) | [`planners.py`](src/argus_autonomy/planners.py) |
| Safety | Geodesic routing around the no-fly zone (no corner cutting). Waypoints are accepted only if the drone can still reach base with reserve; return-to-base is forced when the margin runs out | [`safety.py`](src/argus_autonomy/safety.py) |

The [technical report](docs/report/report.pdf) gives the equations, the experimental protocol and the failure analysis in full.

## Run it

```bash
pip install -e ".[dev]"          # Python ≥ 3.10; numpy, scipy (+ matplotlib, imageio for figures)
pytest -q                        # 23 tests: Bayes update, monotonicity, geofence, RTB, metrics

argus-demo --planner info-rate --seed 47 --gif mission.gif
argus-demo --planner prior-lawnmower --condition degraded-prior --seed 68

python experiments/run_benchmark.py --seeds 100          # ~1 min on 2 cores → results/, media/
python experiments/calibrate_prediction.py --seeds 50
python experiments/ablation_prior_tempering.py --seeds 100
python experiments/make_media.py
```

Every number in this README and in the report comes from the files these scripts write. Results are deterministic for a given seed.

## Scope and honest limitations

- **2-D grid world.** No vehicle dynamics, wind on the aircraft, terrain, smoke occlusion or communication loss.
- **Independent per-cell sensor noise.** Real thermal false alarms (sun glint, hot rock) are spatially correlated.
- **Toy fire model.** It is not Rothermel/FARSITE, and the spread parameters are not fitted to real fire data.
- **Simple safety monitor.** It is a checkable runtime invariant on a grid, not a formal reachability guarantee for continuous dynamics.
- **One vehicle and a greedy one-step planner.** There is no multi-step lookahead and no multi-UAV coordination.

## Citation

See [`CITATION.cff`](CITATION.cff). Releases are archived on Zenodo with a DOI.

## License

Apache-2.0, see [`LICENSE`](LICENSE).
