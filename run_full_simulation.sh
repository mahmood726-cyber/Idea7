#!/bin/bash
# Script to run full extended simulation study (20,000 simulations)
#
# This will take 24-48 hours to complete on a machine with 8+ cores
# 
# Usage:
#   ./run_full_simulation.sh

set -e

echo "================================================================================"
echo "EXTENDED SIMULATION STUDY - FULL RUN"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - Scenarios: 20"
echo "  - Replications per scenario: 1000"
echo "  - Total simulations: 20,000"
echo "  - CPU cores: 8 (adjust with --n_cores if needed)"
echo ""
echo "Estimated runtime: 24-48 hours"
echo ""
echo "Results will be saved to: simulations/results/"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p simulations/results
mkdir -p simulations/logs

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="simulations/logs/full_simulation_${TIMESTAMP}.log"

echo "Starting simulation at: $(date)"
echo "Logging to: ${LOGFILE}"
echo ""

# Run simulation
python3 simulations/extended_simulation_study.py \
    --n_reps 1000 \
    --n_cores 8 \
    2>&1 | tee "${LOGFILE}"

echo ""
echo "================================================================================"
echo "SIMULATION COMPLETE"
echo "================================================================================"
echo ""
echo "Completed at: $(date)"
echo "Log file: ${LOGFILE}"
echo "Results: simulations/results/extended_simulation_raw_results.csv"
echo "Metrics: simulations/results/extended_simulation_metrics.csv"
echo ""
echo "Next steps:"
echo "  1. Verify results: python3 -c 'import pandas as pd; df=pd.read_csv(\"simulations/results/extended_simulation_raw_results.csv\"); print(f\"Total: {len(df)} rows\")'"
echo "  2. Commit results: git add simulations/results/ && git commit -m 'Add full simulation results (20k)' && git push"
echo "  3. Update manuscript with results"
echo ""
