
### Headless Safety Benchmark Suite

The headless benchmark suite compares exploration and attitude-control choices
on four repeatable safety scenarios:

* **Indoor stress:** three robots, walls, known obstacles, and hidden obstacles.
* **Blind corner:** a hidden obstacle immediately beyond an occluding turn.
* **Decentralized hidden obstacle:** two robots acquire different local obstacle knowledge.
* **Small-FoV zigzag:** repeated turns and hidden obstacles stress blind-side motion.

Every scenario can be evaluated with both `Frontier` and `CoScan` exploration
and these attitude policies:

* `simple` - constant yaw rate.
* `visibility_area` - visibility-promoting yaw.
* `velocity_tracking_yaw` - aligns the sensor with robot velocity.
* `gatekeeper` - switches between visibility-promoting and velocity-tracking yaw.

The suite runs without Matplotlib or an Isaac viewport. From the project
container, run the complete 32-case matrix with:

```bash
python3 examples/benchmark_config_suite.py
```

Run a short Gatekeeper-versus-visibility smoke test with:

```bash
python3 examples/benchmark_config_suite.py \
  --scenario blind_corner \
  --algorithm Frontier \
  --attitude visibility_area \
  --attitude gatekeeper \
  --max-steps 20 \
  --output-dir output/headless_safety_smoke
```

Results are written to `output/headless_safety_benchmark/` as:

* `raw_results.csv` - one row per run, including seed and termination details.
* `summary.csv` - rates and averages grouped by scenario, algorithm, and policy.
* `summary.md` - the same summary as a readable Markdown table.

The main metrics are success rate, physical collision rate, optimizer
infeasibility rate, timeout rate, mean visibility violations per robot,
coverage, and completion time over successful runs. Collisions are deliberately
reported separately from solver infeasibility so a numerical controller failure
is not mislabeled as physical contact.