#!/usr/bin/env bash
set -euo pipefail

output=$(julia --project --startup-file=no --color=no test/sde.jl 2>&1)
echo "$output"

correlation=$(echo "$output" | grep -o '\[ Info: [0-9.]*' | grep -o '[0-9.]*' | tail -1)
echo "METRIC denoised_correlation=$correlation"
