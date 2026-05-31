#!/usr/bin/env bash
set -euo pipefail

julia --project --startup-file=no --color=no autoresearch.jl 2>/dev/null
