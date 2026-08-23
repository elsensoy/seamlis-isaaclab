#!/bin/bash
# run_comparison.sh
set -euo pipefail

# 1. Define the absolute path to the Isaac Sim python runner
ISAAC_PYTHON="/workspace/isaaclab/_isaac_sim/python.sh"

# 2. Define the script 
RUN_SCRIPT="scripts/run_isaac.py"

CONFIGS=(
  "configs/test_1_blind_corner.yaml"
)

for CFG in "${CONFIGS[@]}"; do
  # Check if config actually exists before trying to run it
  if [ ! -f "$CFG" ]; then
    echo "Error: Config $CFG not found. Skipping..."
    continue
  fi

  echo "------------------------------------------------"
  echo "Running Experiment: $CFG"
  
  # Added the 'scripts/' prefix to the python call
  OMNI_KIT_ALLOW_ROOT=1 $ISAAC_PYTHON $RUN_SCRIPT \
    -c "$CFG" \
    --explore-ui \
    --viz kit
done

echo "Comparison complete. Check logs/comparison_results.csv"