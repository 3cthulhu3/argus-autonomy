# Screening CLEAR rules (seeds 5000+, disjoint from the benchmark)

Errors: B = false ALERT, A/C = real fire CLEARed. Times: B = median time to CLEAR, A/C = median time to correct ALERT (min).

| rule | B false alerts | B undecided | B t_clear | A fires missed | A undecided | A t_alert | C fires missed | C t_alert |
|---|---|---|---|---|---|---|---|---|
| M1 (milestone i) | 26/1000 | 0 | 4.33 | 14/500 | 0 | 2.17 | 1/300 | 2.17 |
| gate: hotspot seen | 26/1000 | 5 | 4.67 | 3/500 | 0 | 2.17 | 1/300 | 2.17 |
| gate: hotspot + ring seen | 26/1000 | 567 | 17.17 | 0/500 | 2 | 2.17 | 1/300 | 2.17 |
| H1 cue split phi = 0.5 | 25/1000 | 70 | 7.67 | 2/500 | 0 | 3.00 | 1/300 | 2.67 |
| phi = 0.5 + hotspot gate | 25/1000 | 72 | 8.00 | 2/500 | 0 | 3.00 | 1/300 | 2.67 |
