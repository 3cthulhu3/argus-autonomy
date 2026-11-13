# v2.1 held-out results — assimilation and regime signal

200 scenarios (seeds 21000–21199), forecasts issued at minutes 20, 30, 40, leads +15 and +30 min; 600 paired forecasts per lead. Configuration: {'rw_wind_ms': 0.8}.

Wind shift in 77 scenarios; truth fire reached the domain edge by minute 70 in 1.

## Verdicts

| | outcome | evidence |
|---|---|---|
| E40 | held | JE < OL on summed squared error at +30 min in 452 of 600 non-tied pairs (p = 5.0e-37) |
| E41 | held | JE < SO at +30 min in 406 of 600 (p = 1.6e-18) |
| E42 | held | SO < OL at +15 min in 492 of 600 (p = 8.9e-60) |
| E43 | held | median wind error JE 0.82 m/s vs prior 1.67 m/s (ratio 0.49; bound 0.60) |
| E44 | held | moisture error JE < prior in 355 of 600 (p = 4.1e-06); medians 1.08 vs 1.36 points |
| E45 | held | JE +30 min: median area ratio 1.04 (bound 0.8-1.25), median envelope cover 0.97 (bound 0.9) |
| E46 | held | |log10 N_c error| JE < prior in 413 of 600 (p = 7.2e-21); medians 0.13 vs 0.26 |
| E47 | held | v2.0 relevant-set Brier ranks SO above JE at +30 min in 363 of 600 pairs |

**8 of 8 held.**

## Forecast skill

Summed squared error over the domain divided by the true burnt area (lower is better; a deterministic forecast's value is its misclassified area over the true area). v2.0's relevant-set Brier and the other v2.0 scores are reported for continuity.

| lead | method | SSE / true area, median (mean) | Brier (v2.0), median | IoU of P ≥ 0.5, median | area ratio, median [IQR] | envelope cover, median |
|---|---|---|---|---|---|---|
| +15 | OL | 0.206 (0.286) | 0.093 | 0.75 | 1.05 [0.89–1.22] | 1.00 |
| +15 | SO | 0.169 (0.213) | 0.088 | 0.81 | 1.07 [0.96–1.19] | 0.99 |
| +15 | JE | 0.145 (0.161) | 0.092 | 0.83 | 1.05 [0.97–1.13] | 0.98 |
| +30 | OL | 0.231 (0.325) | 0.099 | 0.73 | 1.04 [0.87–1.23] | 0.99 |
| +30 | SO | 0.199 (0.265) | 0.097 | 0.76 | 1.07 [0.95–1.21] | 0.98 |
| +30 | JE | 0.159 (0.184) | 0.101 | 0.81 | 1.04 [0.94–1.13] | 0.97 |

Paired sign tests (wins : losses, one-sided p):

| comparison | SSE +15 | SSE +30 | Brier (v2.0) +15 | Brier (v2.0) +30 |
|---|---|---|---|---|
| JE < OL | 465:135 (1e-43) | 452:148 (5e-37) | 277:323 (1e+00) | 260:340 (1e+00) |
| JE < SO | 402:197 (2e-17) | 406:194 (2e-18) | 242:358 (1e+00) | 237:363 (1e+00) |
| SO < OL | 492:108 (9e-60) | 465:135 (1e-43) | 363:237 (2e-07) | 357:243 (2e-06) |

Median SSE reduction of JE: JE vs OL +15 30 %; JE vs SO +15 14 %; JE vs OL +30 31 %; JE vs SO +30 20 %.

### By wind shift (post hoc, not tested)

| scenarios | n | JE < SO, SSE +30 | JE < OL, SSE +30 | SO < OL, SSE +30 |
|---|---|---|---|---|
| no shift | 123 | 268:101 (7e-19) | 290:79 (1e-29) | 304:65 (2e-38) |
| shift | 77 | 138:93 (2e-03) | 162:69 (4e-10) | 161:70 (1e-09) |

## Inputs

| method | wind error, median (m/s) | moisture error, median (points) | log α at issue, 10–90 % |
|---|---|---|---|
| OL | 1.67 | 1.36 | -0.06 to 0.07 |
| SO | 1.67 | 1.36 | -0.06 to 0.07 |
| JE | 0.82 | 1.08 | -0.36 to 0.25 |

## Regime signal

True regime at the issue times: wind-driven 549, transitional 51, plume-dominated 0 (of 600).

| method | median abs. log10 N_c error | regime Brier, mean | argmax class correct | lights G / A / R |
|---|---|---|---|---|
| OL | 0.26 | 0.119 | 90.7 % | 505 / 77 / 18 |
| SO | 0.23 | 0.118 | 91.2 % | 501 / 77 / 22 |
| JE | 0.13 | 0.082 | 93.3 % | 526 / 56 / 18 |
