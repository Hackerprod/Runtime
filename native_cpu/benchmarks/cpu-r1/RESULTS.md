# CPU R1 Benchmark Report

Offline summary of recorded benchmark evidence. Timing is wall-clock data; worker times overlap and are not summed.

## Conversation

| Case | Family | Workload | Profile | Prefill median ± stdev s | Native decode median ± stdev s | Sampling median ± stdev s | Turn sum median ± stdev s | Conversation wall median ± stdev s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| attribution-R0-R1-historical | attribution | historical | R0 | 18.922 ± 1.585 | 8.841 ± 0.677 | 0.399 ± 0.016 | 28.063 ± 2.213 | 28.063 ± 2.213 |
| attribution-R0-R1-historical | attribution | historical | R1 | 17.274 ± 1.624 | 8.477 ± 0.704 | 0.391 ± 0.018 | 26.150 ± 2.336 | 26.150 ± 2.336 |
| attribution-R0-R1-second | attribution | second | R0 | 1.663 ± 0.085 | 0.757 ± 0.012 | 0.051 ± 0.001 | 2.481 ± 0.089 | 2.481 ± 0.089 |
| attribution-R0-R1-second | attribution | second | R1 | 1.526 ± 0.050 | 0.748 ± 0.025 | 0.050 ± 0.002 | 2.322 ± 0.076 | 2.322 ± 0.076 |
| attribution-R0-R2-historical | attribution | historical | R0 | 18.117 ± 0.238 | 8.426 ± 0.102 | 0.390 ± 0.003 | 26.924 ± 0.316 | 26.924 ± 0.316 |
| attribution-R0-R2-historical | attribution | historical | R2 | 1.763 ± 0.007 | 8.386 ± 0.067 | 0.387 ± 0.003 | 10.549 ± 0.071 | 10.549 ± 0.071 |
| attribution-R0-R2-second | attribution | second | R0 | 1.637 ± 0.015 | 0.746 ± 0.004 | 0.050 ± 0.001 | 2.435 ± 0.014 | 2.435 ± 0.014 |
| attribution-R0-R2-second | attribution | second | R2 | 0.710 ± 0.005 | 0.747 ± 0.010 | 0.051 ± 0.001 | 1.514 ± 0.011 | 1.514 ± 0.011 |
| attribution-R0-R3-historical | attribution | historical | R0 | 18.142 ± 0.918 | 8.368 ± 0.750 | 0.390 ± 0.026 | 26.926 ± 1.694 | 26.926 ± 1.694 |
| attribution-R0-R3-historical | attribution | historical | R3 | 1.694 ± 0.086 | 8.394 ± 0.648 | 0.390 ± 0.021 | 10.533 ± 0.730 | 10.533 ± 0.730 |
| attribution-R0-R3-second | attribution | second | R0 | 1.725 ± 0.026 | 0.779 ± 0.013 | 0.057 ± 0.001 | 2.573 ± 0.032 | 2.573 ± 0.032 |
| attribution-R0-R3-second | attribution | second | R3 | 0.686 ± 0.079 | 0.781 ± 0.108 | 0.057 ± 0.001 | 1.519 ± 0.184 | 1.519 ± 0.184 |
| scaling-S1-S0-historical | scaling | historical | S1 | 1.726 ± 0.024 | 8.815 ± 0.174 | 0.424 ± 0.008 | 10.995 ± 0.180 | 10.996 ± 0.180 |
| scaling-S1-S0-historical | scaling | historical | S0 | 1.852 ± 0.026 | 9.485 ± 0.080 | 0.477 ± 0.007 | 11.826 ± 0.111 | 11.826 ± 0.111 |
| scaling-S1-S0-second | scaling | second | S1 | 0.683 ± 0.007 | 0.779 ± 0.057 | 0.055 ± 0.002 | 1.521 ± 0.063 | 1.521 ± 0.063 |
| scaling-S1-S0-second | scaling | second | S0 | 0.706 ± 0.034 | 0.797 ± 0.004 | 0.061 ± 0.001 | 1.566 ± 0.037 | 1.566 ± 0.037 |
| scaling-S1-P2-historical | scaling | historical | S1 | 1.731 ± 0.058 | 8.824 ± 0.081 | 0.428 ± 0.003 | 10.999 ± 0.113 | 10.999 ± 0.113 |
| scaling-S1-P2-historical | scaling | historical | P2 | 1.704 ± 0.077 | 8.480 ± 0.208 | 0.424 ± 0.004 | 10.630 ± 0.278 | 10.631 ± 0.278 |
| scaling-S1-P2-second | scaling | second | S1 | 0.692 ± 0.026 | 0.774 ± 0.017 | 0.055 ± 0.001 | 1.542 ± 0.029 | 1.542 ± 0.029 |
| scaling-S1-P2-second | scaling | second | P2 | 0.682 ± 0.029 | 0.746 ± 0.045 | 0.058 ± 0.003 | 1.494 ± 0.071 | 1.494 ± 0.071 |
| scaling-S1-P4-historical | scaling | historical | S1 | 1.776 ± 0.136 | 9.042 ± 1.341 | 0.442 ± 0.047 | 11.279 ± 1.522 | 11.279 ± 1.522 |
| scaling-S1-P4-historical | scaling | historical | P4 | 1.937 ± 0.404 | 9.703 ± 1.222 | 0.466 ± 0.038 | 12.139 ± 1.651 | 12.139 ± 1.651 |
| scaling-S1-P4-second | scaling | second | S1 | 0.684 ± 0.005 | 0.778 ± 0.018 | 0.056 ± 0.002 | 1.521 ± 0.019 | 1.521 ± 0.019 |
| scaling-S1-P4-second | scaling | second | P4 | 0.710 ± 0.018 | 0.797 ± 0.020 | 0.058 ± 0.002 | 1.570 ± 0.020 | 1.570 ± 0.020 |

## Forced contexts

Each value is median ± sample standard deviation within its own six-pair comparison.

| Case | Context | Profile | Prefill s | Native decode s | Prefill tok/s | Decode tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| forced-S1-S0-64 | 64 | S1 | 0.420 ± 0.012 | 0.247 ± 0.004 | 152.433 ± 4.261 | 129.554 ± 2.328 |
| forced-S1-S0-64 | 64 | S0 | 0.434 ± 0.013 | 0.252 ± 0.026 | 147.316 ± 4.300 | 126.904 ± 10.818 |
| forced-S1-S0-256 | 256 | S1 | 1.919 ± 0.117 | 0.303 ± 0.010 | 133.428 ± 7.160 | 105.726 ± 3.352 |
| forced-S1-S0-256 | 256 | S0 | 2.003 ± 0.025 | 0.322 ± 0.002 | 127.825 ± 1.587 | 99.315 ± 0.510 |
| forced-S1-S0-1024 | 1024 | S1 | 11.434 ± 0.171 | 0.541 ± 0.004 | 89.566 ± 1.331 | 59.134 ± 0.393 |
| forced-S1-S0-1024 | 1024 | S0 | 12.602 ± 0.171 | 0.607 ± 0.036 | 81.257 ± 1.077 | 52.760 ± 2.743 |
| forced-S1-S0-1792 | 1792 | S1 | 25.091 ± 1.453 | 0.743 ± 0.038 | 71.419 ± 3.632 | 43.062 ± 2.034 |
| forced-S1-S0-1792 | 1792 | S0 | 25.537 ± 0.088 | 0.754 ± 0.007 | 70.173 ± 0.242 | 42.451 ± 0.386 |
| forced-S1-P2-64 | 64 | S1 | 0.409 ± 0.006 | 0.235 ± 0.002 | 156.453 ± 2.375 | 135.946 ± 1.403 |
| forced-S1-P2-64 | 64 | P2 | 0.405 ± 0.004 | 0.234 ± 0.007 | 158.197 ± 1.460 | 136.539 ± 3.647 |
| forced-S1-P2-256 | 256 | S1 | 1.846 ± 0.005 | 0.290 ± 0.002 | 138.650 ± 0.352 | 110.258 ± 0.821 |
| forced-S1-P2-256 | 256 | P2 | 1.807 ± 0.027 | 0.284 ± 0.002 | 141.699 ± 2.043 | 112.782 ± 0.882 |
| forced-S1-P2-1024 | 1024 | S1 | 10.948 ± 0.149 | 0.514 ± 0.003 | 93.534 ± 1.252 | 62.314 ± 0.346 |
| forced-S1-P2-1024 | 1024 | P2 | 10.699 ± 0.094 | 0.506 ± 0.006 | 95.708 ± 0.829 | 63.271 ± 0.724 |
| forced-S1-P2-1792 | 1792 | S1 | 25.026 ± 0.203 | 0.741 ± 0.008 | 71.609 ± 0.581 | 43.210 ± 0.454 |
| forced-S1-P2-1792 | 1792 | P2 | 24.053 ± 0.181 | 0.714 ± 0.013 | 74.501 ± 0.561 | 44.850 ± 0.793 |
| forced-S1-P4-64 | 64 | S1 | 0.387 ± 0.002 | 0.224 ± 0.001 | 165.434 ± 0.999 | 143.137 ± 0.845 |
| forced-S1-P4-64 | 64 | P4 | 0.399 ± 0.008 | 0.231 ± 0.007 | 160.324 ± 3.100 | 138.753 ± 3.782 |
| forced-S1-P4-256 | 256 | S1 | 1.771 ± 0.015 | 0.277 ± 0.005 | 144.569 ± 1.226 | 115.675 ± 2.002 |
| forced-S1-P4-256 | 256 | P4 | 1.799 ± 0.027 | 0.285 ± 0.059 | 142.292 ± 2.153 | 112.351 ± 17.172 |
| forced-S1-P4-1024 | 1024 | S1 | 11.238 ± 1.671 | 0.531 ± 0.111 | 91.178 ± 10.891 | 60.336 ± 9.382 |
| forced-S1-P4-1024 | 1024 | P4 | 12.769 ± 2.540 | 0.612 ± 0.128 | 80.569 ± 15.152 | 52.826 ± 10.569 |
| forced-S1-P4-1792 | 1792 | S1 | 25.129 ± 0.292 | 0.746 ± 0.005 | 71.314 ± 0.820 | 42.897 ± 0.306 |
| forced-S1-P4-1792 | 1792 | P4 | 26.530 ± 2.000 | 0.769 ± 0.042 | 67.631 ± 5.086 | 41.668 ± 2.290 |

## Diagnostics (milliseconds)

Participant compute and controller wait are separate. Participant work overlaps and MUST NOT be added to elapsed time.

| Profile | Phase | QKV | Attention | Output | FFN | Vocab | Remaining | Participant: compute ms | Controller wait |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2-off | decode | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0: 0.000, 1: 0.000 | 0.000 |
| P2-off | prefill | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0: 0.000, 1: 0.000 | 0.000 |
| P2-on | decode | 31.332 | 67.689 | 16.119 | 145.339 | 15.792 | 3.805 | 0: 207.796, 1: 174.728 | 0.841 |
| P2-on | prefill | 247.353 | 247.828 | 126.171 | 1159.620 | 0.484 | 28.184 | 0: 1526.945, 1: 1286.884 | 6.112 |
| P4-off | decode | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0: 0.000, 1: 0.000, 2: 0.000, 3: 0.000 | 0.000 |
| P4-off | prefill | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0: 0.000, 1: 0.000, 2: 0.000, 3: 0.000 | 0.000 |
| P4-on | decode | 35.010 | 76.523 | 18.448 | 160.245 | 16.556 | 4.983 | 0: 210.500, 1: 204.947, 2: 196.549, 3: 195.488 | 21.177 |
| P4-on | prefill | 290.029 | 306.390 | 153.648 | 1297.888 | 0.477 | 41.452 | 0: 1548.146, 1: 1509.378, 2: 1444.630, 3: 1436.635 | 192.165 |
| S1-off | decode | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0: 0.000 | 0.000 |
| S1-off | prefill | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0: 0.000 | 0.000 |
| S1-on | decode | 31.423 | 71.970 | 15.938 | 149.178 | 16.028 | 3.757 | 0: 212.712 | 0.042 |
| S1-on | prefill | 250.129 | 262.562 | 125.275 | 1180.501 | 0.463 | 28.482 | 0: 1555.023 | 0.322 |

### Profiling overhead (paired on − off)

Negative estimates and wide dispersion are measurement noise, not evidence that profiling improves performance. No pairs were discarded.

| Profile | Median overhead ms | Stdev ms | Min ms | Max ms | Median wall reduction % |
| --- | --- | --- | --- | --- | --- |
| S1 | -2.960 | 382.657 | -937.001 | 54.748 | 0.14 |
| P2 | 6.021 | 23.133 | -1.415 | 60.124 | -0.29 |
| P4 | -33.072 | 168.069 | -140.436 | 325.862 | 1.37 |

## Paired reduction distributions (per comparison; never pooled)

| Case | Median % | Stdev % | Min % | Max % |
| --- | --- | --- | --- | --- |
| attribution-R0-R1-historical | 5.703 | 2.297 | 4.996 | 11.051 |
| attribution-R0-R1-second | 4.368 | 4.733 | 0.364 | 13.437 |
| attribution-R0-R2-historical | 60.880 | 0.293 | 60.669 | 61.402 |
| attribution-R0-R2-second | 37.921 | 0.703 | 36.702 | 38.583 |
| attribution-R0-R3-historical | 61.202 | 3.351 | 54.469 | 64.745 |
| attribution-R0-R3-second | 40.031 | 6.914 | 23.845 | 42.004 |
| scaling-S1-S0-historical | -8.600 | 2.010 | -9.346 | -4.126 |
| scaling-S1-S0-second | -2.870 | 4.547 | -9.136 | 5.029 |
| scaling-S1-P2-historical | 3.259 | 2.684 | -2.459 | 5.072 |
| scaling-S1-P2-second | 1.490 | 4.803 | -5.862 | 7.130 |
| scaling-S1-P4-historical | -5.138 | 3.857 | -8.432 | 0.340 |
| scaling-S1-P4-second | -3.212 | 2.399 | -5.924 | 0.877 |

## Historical canonical prompt (pair 1)

Each cell is `prompt_tokens_total / prefix_tokens_reused / prefill_tokens_evaluated / lm_head_calls`.

| Turn | R0–R3 |
| --- | --- |
| 1 | R0: 63/0/63/{'prefill': 63, 'decode': 21}<br>R1: 63/0/63/{'prefill': 1, 'decode': 21}<br>R2: 63/0/63/{'prefill': 63, 'decode': 21}<br>R3: 63/0/63/{'prefill': 1, 'decode': 21} |
| 2 | R0: 110/0/110/{'prefill': 110, 'decode': 213}<br>R1: 110/0/110/{'prefill': 1, 'decode': 213}<br>R2: 110/84/26/{'prefill': 26, 'decode': 213}<br>R3: 110/84/26/{'prefill': 1, 'decode': 213} |
| 3 | R0: 350/0/350/{'prefill': 350, 'decode': 32}<br>R1: 350/0/350/{'prefill': 1, 'decode': 32}<br>R2: 350/323/27/{'prefill': 27, 'decode': 32}<br>R3: 350/323/27/{'prefill': 1, 'decode': 32} |
| 4 | R0: 403/0/403/{'prefill': 403, 'decode': 0}<br>R1: 403/0/403/{'prefill': 1, 'decode': 0}<br>R2: 403/382/21/{'prefill': 21, 'decode': 0}<br>R3: 403/382/21/{'prefill': 1, 'decode': 0} |
| 5 | R0: 430/0/430/{'prefill': 430, 'decode': 255}<br>R1: 430/0/430/{'prefill': 1, 'decode': 255}<br>R2: 430/403/27/{'prefill': 27, 'decode': 255}<br>R3: 430/403/27/{'prefill': 1, 'decode': 255} |
| 6 | R0: 715/0/715/{'prefill': 715, 'decode': 212}<br>R1: 715/0/715/{'prefill': 1, 'decode': 212}<br>R2: 715/685/30/{'prefill': 30, 'decode': 212}<br>R3: 715/685/30/{'prefill': 1, 'decode': 212} |

## Raw evidence

- [attribution-R0-R1-historical-pair01.json](results/attribution-R0-R1-historical-pair01.json)
- [attribution-R0-R1-historical-pair02.json](results/attribution-R0-R1-historical-pair02.json)
- [attribution-R0-R1-historical-pair03.json](results/attribution-R0-R1-historical-pair03.json)
- [attribution-R0-R1-historical-pair04.json](results/attribution-R0-R1-historical-pair04.json)
- [attribution-R0-R1-historical-pair05.json](results/attribution-R0-R1-historical-pair05.json)
- [attribution-R0-R1-historical-pair06.json](results/attribution-R0-R1-historical-pair06.json)
- [attribution-R0-R1-historical-summary.json](results/attribution-R0-R1-historical-summary.json)
- [attribution-R0-R1-second-pair01.json](results/attribution-R0-R1-second-pair01.json)
- [attribution-R0-R1-second-pair02.json](results/attribution-R0-R1-second-pair02.json)
- [attribution-R0-R1-second-pair03.json](results/attribution-R0-R1-second-pair03.json)
- [attribution-R0-R1-second-pair04.json](results/attribution-R0-R1-second-pair04.json)
- [attribution-R0-R1-second-pair05.json](results/attribution-R0-R1-second-pair05.json)
- [attribution-R0-R1-second-pair06.json](results/attribution-R0-R1-second-pair06.json)
- [attribution-R0-R1-second-summary.json](results/attribution-R0-R1-second-summary.json)
- [attribution-R0-R2-historical-pair01.json](results/attribution-R0-R2-historical-pair01.json)
- [attribution-R0-R2-historical-pair02.json](results/attribution-R0-R2-historical-pair02.json)
- [attribution-R0-R2-historical-pair03.json](results/attribution-R0-R2-historical-pair03.json)
- [attribution-R0-R2-historical-pair04.json](results/attribution-R0-R2-historical-pair04.json)
- [attribution-R0-R2-historical-pair05.json](results/attribution-R0-R2-historical-pair05.json)
- [attribution-R0-R2-historical-pair06.json](results/attribution-R0-R2-historical-pair06.json)
- [attribution-R0-R2-historical-summary.json](results/attribution-R0-R2-historical-summary.json)
- [attribution-R0-R2-second-pair01.json](results/attribution-R0-R2-second-pair01.json)
- [attribution-R0-R2-second-pair02.json](results/attribution-R0-R2-second-pair02.json)
- [attribution-R0-R2-second-pair03.json](results/attribution-R0-R2-second-pair03.json)
- [attribution-R0-R2-second-pair04.json](results/attribution-R0-R2-second-pair04.json)
- [attribution-R0-R2-second-pair05.json](results/attribution-R0-R2-second-pair05.json)
- [attribution-R0-R2-second-pair06.json](results/attribution-R0-R2-second-pair06.json)
- [attribution-R0-R2-second-summary.json](results/attribution-R0-R2-second-summary.json)
- [attribution-R0-R3-historical-pair01.json](results/attribution-R0-R3-historical-pair01.json)
- [attribution-R0-R3-historical-pair02.json](results/attribution-R0-R3-historical-pair02.json)
- [attribution-R0-R3-historical-pair03.json](results/attribution-R0-R3-historical-pair03.json)
- [attribution-R0-R3-historical-pair04.json](results/attribution-R0-R3-historical-pair04.json)
- [attribution-R0-R3-historical-pair05.json](results/attribution-R0-R3-historical-pair05.json)
- [attribution-R0-R3-historical-pair06.json](results/attribution-R0-R3-historical-pair06.json)
- [attribution-R0-R3-historical-summary.json](results/attribution-R0-R3-historical-summary.json)
- [attribution-R0-R3-second-pair01.json](results/attribution-R0-R3-second-pair01.json)
- [attribution-R0-R3-second-pair02.json](results/attribution-R0-R3-second-pair02.json)
- [attribution-R0-R3-second-pair03.json](results/attribution-R0-R3-second-pair03.json)
- [attribution-R0-R3-second-pair04.json](results/attribution-R0-R3-second-pair04.json)
- [attribution-R0-R3-second-pair05.json](results/attribution-R0-R3-second-pair05.json)
- [attribution-R0-R3-second-pair06.json](results/attribution-R0-R3-second-pair06.json)
- [attribution-R0-R3-second-summary.json](results/attribution-R0-R3-second-summary.json)
- [diagnostics-P2-pair01.json](results/diagnostics-P2-pair01.json)
- [diagnostics-P2-pair02.json](results/diagnostics-P2-pair02.json)
- [diagnostics-P2-pair03.json](results/diagnostics-P2-pair03.json)
- [diagnostics-P2-pair04.json](results/diagnostics-P2-pair04.json)
- [diagnostics-P2-pair05.json](results/diagnostics-P2-pair05.json)
- [diagnostics-P2-pair06.json](results/diagnostics-P2-pair06.json)
- [diagnostics-P2-summary.json](results/diagnostics-P2-summary.json)
- [diagnostics-P4-pair01.json](results/diagnostics-P4-pair01.json)
- [diagnostics-P4-pair02.json](results/diagnostics-P4-pair02.json)
- [diagnostics-P4-pair03.json](results/diagnostics-P4-pair03.json)
- [diagnostics-P4-pair04.json](results/diagnostics-P4-pair04.json)
- [diagnostics-P4-pair05.json](results/diagnostics-P4-pair05.json)
- [diagnostics-P4-pair06.json](results/diagnostics-P4-pair06.json)
- [diagnostics-P4-summary.json](results/diagnostics-P4-summary.json)
- [diagnostics-S1-pair01.json](results/diagnostics-S1-pair01.json)
- [diagnostics-S1-pair02.json](results/diagnostics-S1-pair02.json)
- [diagnostics-S1-pair03.json](results/diagnostics-S1-pair03.json)
- [diagnostics-S1-pair04.json](results/diagnostics-S1-pair04.json)
- [diagnostics-S1-pair05.json](results/diagnostics-S1-pair05.json)
- [diagnostics-S1-pair06.json](results/diagnostics-S1-pair06.json)
- [diagnostics-S1-summary.json](results/diagnostics-S1-summary.json)
- [forced-S1-P2-1024-pair01.json](results/forced-S1-P2-1024-pair01.json)
- [forced-S1-P2-1024-pair02.json](results/forced-S1-P2-1024-pair02.json)
- [forced-S1-P2-1024-pair03.json](results/forced-S1-P2-1024-pair03.json)
- [forced-S1-P2-1024-pair04.json](results/forced-S1-P2-1024-pair04.json)
- [forced-S1-P2-1024-pair05.json](results/forced-S1-P2-1024-pair05.json)
- [forced-S1-P2-1024-pair06.json](results/forced-S1-P2-1024-pair06.json)
- [forced-S1-P2-1024-summary.json](results/forced-S1-P2-1024-summary.json)
- [forced-S1-P2-1792-pair01.json](results/forced-S1-P2-1792-pair01.json)
- [forced-S1-P2-1792-pair02.json](results/forced-S1-P2-1792-pair02.json)
- [forced-S1-P2-1792-pair03.json](results/forced-S1-P2-1792-pair03.json)
- [forced-S1-P2-1792-pair04.json](results/forced-S1-P2-1792-pair04.json)
- [forced-S1-P2-1792-pair05.json](results/forced-S1-P2-1792-pair05.json)
- [forced-S1-P2-1792-pair06.json](results/forced-S1-P2-1792-pair06.json)
- [forced-S1-P2-1792-summary.json](results/forced-S1-P2-1792-summary.json)
- [forced-S1-P2-256-pair01.json](results/forced-S1-P2-256-pair01.json)
- [forced-S1-P2-256-pair02.json](results/forced-S1-P2-256-pair02.json)
- [forced-S1-P2-256-pair03.json](results/forced-S1-P2-256-pair03.json)
- [forced-S1-P2-256-pair04.json](results/forced-S1-P2-256-pair04.json)
- [forced-S1-P2-256-pair05.json](results/forced-S1-P2-256-pair05.json)
- [forced-S1-P2-256-pair06.json](results/forced-S1-P2-256-pair06.json)
- [forced-S1-P2-256-summary.json](results/forced-S1-P2-256-summary.json)
- [forced-S1-P2-64-pair01.json](results/forced-S1-P2-64-pair01.json)
- [forced-S1-P2-64-pair02.json](results/forced-S1-P2-64-pair02.json)
- [forced-S1-P2-64-pair03.json](results/forced-S1-P2-64-pair03.json)
- [forced-S1-P2-64-pair04.json](results/forced-S1-P2-64-pair04.json)
- [forced-S1-P2-64-pair05.json](results/forced-S1-P2-64-pair05.json)
- [forced-S1-P2-64-pair06.json](results/forced-S1-P2-64-pair06.json)
- [forced-S1-P2-64-summary.json](results/forced-S1-P2-64-summary.json)
- [forced-S1-P4-1024-pair01.json](results/forced-S1-P4-1024-pair01.json)
- [forced-S1-P4-1024-pair02.json](results/forced-S1-P4-1024-pair02.json)
- [forced-S1-P4-1024-pair03.json](results/forced-S1-P4-1024-pair03.json)
- [forced-S1-P4-1024-pair04.json](results/forced-S1-P4-1024-pair04.json)
- [forced-S1-P4-1024-pair05.json](results/forced-S1-P4-1024-pair05.json)
- [forced-S1-P4-1024-pair06.json](results/forced-S1-P4-1024-pair06.json)
- [forced-S1-P4-1024-summary.json](results/forced-S1-P4-1024-summary.json)
- [forced-S1-P4-1792-pair01.json](results/forced-S1-P4-1792-pair01.json)
- [forced-S1-P4-1792-pair02.json](results/forced-S1-P4-1792-pair02.json)
- [forced-S1-P4-1792-pair03.json](results/forced-S1-P4-1792-pair03.json)
- [forced-S1-P4-1792-pair04.json](results/forced-S1-P4-1792-pair04.json)
- [forced-S1-P4-1792-pair05.json](results/forced-S1-P4-1792-pair05.json)
- [forced-S1-P4-1792-pair06.json](results/forced-S1-P4-1792-pair06.json)
- [forced-S1-P4-1792-summary.json](results/forced-S1-P4-1792-summary.json)
- [forced-S1-P4-256-pair01.json](results/forced-S1-P4-256-pair01.json)
- [forced-S1-P4-256-pair02.json](results/forced-S1-P4-256-pair02.json)
- [forced-S1-P4-256-pair03.json](results/forced-S1-P4-256-pair03.json)
- [forced-S1-P4-256-pair04.json](results/forced-S1-P4-256-pair04.json)
- [forced-S1-P4-256-pair05.json](results/forced-S1-P4-256-pair05.json)
- [forced-S1-P4-256-pair06.json](results/forced-S1-P4-256-pair06.json)
- [forced-S1-P4-256-summary.json](results/forced-S1-P4-256-summary.json)
- [forced-S1-P4-64-pair01.json](results/forced-S1-P4-64-pair01.json)
- [forced-S1-P4-64-pair02.json](results/forced-S1-P4-64-pair02.json)
- [forced-S1-P4-64-pair03.json](results/forced-S1-P4-64-pair03.json)
- [forced-S1-P4-64-pair04.json](results/forced-S1-P4-64-pair04.json)
- [forced-S1-P4-64-pair05.json](results/forced-S1-P4-64-pair05.json)
- [forced-S1-P4-64-pair06.json](results/forced-S1-P4-64-pair06.json)
- [forced-S1-P4-64-summary.json](results/forced-S1-P4-64-summary.json)
- [forced-S1-S0-1024-pair01.json](results/forced-S1-S0-1024-pair01.json)
- [forced-S1-S0-1024-pair02.json](results/forced-S1-S0-1024-pair02.json)
- [forced-S1-S0-1024-pair03.json](results/forced-S1-S0-1024-pair03.json)
- [forced-S1-S0-1024-pair04.json](results/forced-S1-S0-1024-pair04.json)
- [forced-S1-S0-1024-pair05.json](results/forced-S1-S0-1024-pair05.json)
- [forced-S1-S0-1024-pair06.json](results/forced-S1-S0-1024-pair06.json)
- [forced-S1-S0-1024-summary.json](results/forced-S1-S0-1024-summary.json)
- [forced-S1-S0-1792-pair01.json](results/forced-S1-S0-1792-pair01.json)
- [forced-S1-S0-1792-pair02.json](results/forced-S1-S0-1792-pair02.json)
- [forced-S1-S0-1792-pair03.json](results/forced-S1-S0-1792-pair03.json)
- [forced-S1-S0-1792-pair04.json](results/forced-S1-S0-1792-pair04.json)
- [forced-S1-S0-1792-pair05.json](results/forced-S1-S0-1792-pair05.json)
- [forced-S1-S0-1792-pair06.json](results/forced-S1-S0-1792-pair06.json)
- [forced-S1-S0-1792-summary.json](results/forced-S1-S0-1792-summary.json)
- [forced-S1-S0-256-pair01.json](results/forced-S1-S0-256-pair01.json)
- [forced-S1-S0-256-pair02.json](results/forced-S1-S0-256-pair02.json)
- [forced-S1-S0-256-pair03.json](results/forced-S1-S0-256-pair03.json)
- [forced-S1-S0-256-pair04.json](results/forced-S1-S0-256-pair04.json)
- [forced-S1-S0-256-pair05.json](results/forced-S1-S0-256-pair05.json)
- [forced-S1-S0-256-pair06.json](results/forced-S1-S0-256-pair06.json)
- [forced-S1-S0-256-summary.json](results/forced-S1-S0-256-summary.json)
- [forced-S1-S0-64-pair01.json](results/forced-S1-S0-64-pair01.json)
- [forced-S1-S0-64-pair02.json](results/forced-S1-S0-64-pair02.json)
- [forced-S1-S0-64-pair03.json](results/forced-S1-S0-64-pair03.json)
- [forced-S1-S0-64-pair04.json](results/forced-S1-S0-64-pair04.json)
- [forced-S1-S0-64-pair05.json](results/forced-S1-S0-64-pair05.json)
- [forced-S1-S0-64-pair06.json](results/forced-S1-S0-64-pair06.json)
- [forced-S1-S0-64-summary.json](results/forced-S1-S0-64-summary.json)
- [scaling-S1-P2-historical-pair01.json](results/scaling-S1-P2-historical-pair01.json)
- [scaling-S1-P2-historical-pair02.json](results/scaling-S1-P2-historical-pair02.json)
- [scaling-S1-P2-historical-pair03.json](results/scaling-S1-P2-historical-pair03.json)
- [scaling-S1-P2-historical-pair04.json](results/scaling-S1-P2-historical-pair04.json)
- [scaling-S1-P2-historical-pair05.json](results/scaling-S1-P2-historical-pair05.json)
- [scaling-S1-P2-historical-pair06.json](results/scaling-S1-P2-historical-pair06.json)
- [scaling-S1-P2-historical-summary.json](results/scaling-S1-P2-historical-summary.json)
- [scaling-S1-P2-second-pair01.json](results/scaling-S1-P2-second-pair01.json)
- [scaling-S1-P2-second-pair02.json](results/scaling-S1-P2-second-pair02.json)
- [scaling-S1-P2-second-pair03.json](results/scaling-S1-P2-second-pair03.json)
- [scaling-S1-P2-second-pair04.json](results/scaling-S1-P2-second-pair04.json)
- [scaling-S1-P2-second-pair05.json](results/scaling-S1-P2-second-pair05.json)
- [scaling-S1-P2-second-pair06.json](results/scaling-S1-P2-second-pair06.json)
- [scaling-S1-P2-second-summary.json](results/scaling-S1-P2-second-summary.json)
- [scaling-S1-P4-historical-pair01.json](results/scaling-S1-P4-historical-pair01.json)
- [scaling-S1-P4-historical-pair02.json](results/scaling-S1-P4-historical-pair02.json)
- [scaling-S1-P4-historical-pair03.json](results/scaling-S1-P4-historical-pair03.json)
- [scaling-S1-P4-historical-pair04.json](results/scaling-S1-P4-historical-pair04.json)
- [scaling-S1-P4-historical-pair05.json](results/scaling-S1-P4-historical-pair05.json)
- [scaling-S1-P4-historical-pair06.json](results/scaling-S1-P4-historical-pair06.json)
- [scaling-S1-P4-historical-summary.json](results/scaling-S1-P4-historical-summary.json)
- [scaling-S1-P4-second-pair01.json](results/scaling-S1-P4-second-pair01.json)
- [scaling-S1-P4-second-pair02.json](results/scaling-S1-P4-second-pair02.json)
- [scaling-S1-P4-second-pair03.json](results/scaling-S1-P4-second-pair03.json)
- [scaling-S1-P4-second-pair04.json](results/scaling-S1-P4-second-pair04.json)
- [scaling-S1-P4-second-pair05.json](results/scaling-S1-P4-second-pair05.json)
- [scaling-S1-P4-second-pair06.json](results/scaling-S1-P4-second-pair06.json)
- [scaling-S1-P4-second-summary.json](results/scaling-S1-P4-second-summary.json)
- [scaling-S1-S0-historical-pair01.json](results/scaling-S1-S0-historical-pair01.json)
- [scaling-S1-S0-historical-pair02.json](results/scaling-S1-S0-historical-pair02.json)
- [scaling-S1-S0-historical-pair03.json](results/scaling-S1-S0-historical-pair03.json)
- [scaling-S1-S0-historical-pair04.json](results/scaling-S1-S0-historical-pair04.json)
- [scaling-S1-S0-historical-pair05.json](results/scaling-S1-S0-historical-pair05.json)
- [scaling-S1-S0-historical-pair06.json](results/scaling-S1-S0-historical-pair06.json)
- [scaling-S1-S0-historical-summary.json](results/scaling-S1-S0-historical-summary.json)
- [scaling-S1-S0-second-pair01.json](results/scaling-S1-S0-second-pair01.json)
- [scaling-S1-S0-second-pair02.json](results/scaling-S1-S0-second-pair02.json)
- [scaling-S1-S0-second-pair03.json](results/scaling-S1-S0-second-pair03.json)
- [scaling-S1-S0-second-pair04.json](results/scaling-S1-S0-second-pair04.json)
- [scaling-S1-S0-second-pair05.json](results/scaling-S1-S0-second-pair05.json)
- [scaling-S1-S0-second-pair06.json](results/scaling-S1-S0-second-pair06.json)
- [scaling-S1-S0-second-summary.json](results/scaling-S1-S0-second-summary.json)
- [summary.json](results/summary.json)
