# Headless safety benchmark

This deterministic regression suite compares one to three robots, Frontier and
CoScan goal allocation, four attitude policies, finite field of view, hidden
obstacles, and success/collision/visibility/time metrics. The four fixed
scenarios make failures repeatable while controller code is being developed.

## Run the full matrix

From the project container:

```bash
python3 examples/benchmark_config_suite.py
```

The runner never creates a Matplotlib figure. It writes:

- `output/headless_safety_benchmark/raw_results.csv`
- `output/headless_safety_benchmark/summary.csv`
- `output/headless_safety_benchmark/summary.md`

## Run a short smoke test

```bash
python3 examples/benchmark_config_suite.py \
  --scenario blind_corner \
  --algorithm Frontier \
  --attitude visibility_area \
  --attitude gatekeeper \
  --max-steps 20 \
  --output-dir output/headless_safety_smoke
```

## Repeat matched trials

All policies use the same seed for a given scenario and repeat:

```bash
python3 examples/benchmark_config_suite.py --repeats 10
```

Repeats reuse each fixed scenario with matched seeds across policies. They are
useful for detecting nondeterminism, but they are not independent environment
samples. Use separately generated maps when statistical environment diversity
is required.

## Metrics

- `collision`: geometric contact with an unknown obstacle or another robot.
- `infeasible`: the positional optimizer failed without confirmed contact.
- `timeout`: the robot-count-specific time limit was reached.
- `success`: 98% coverage was reached before collision or timeout.
- `violations_per_robot`: mean unknown-region overlap events per robot.
- `mean_time_success_s`: exploration time over successful trials only.

The separate collision and infeasibility columns are intentional. Combining
them would inflate collision rates when an MPC solve fails numerically.

## Validate the files

```bash
python3 -m unittest discover -s benchmark_tests -p 'test_*benchmark.py' -v
```
