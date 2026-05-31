#!/bin/bash
set -euo pipefail
label="${FM_TS_LABEL:-run_$(date +%Y%m%d_%H%M%S)}"
AUTORESEARCH=1 FM_TS_LABEL="$label" julia --project=. test/flow_matching_timeseries.jl
