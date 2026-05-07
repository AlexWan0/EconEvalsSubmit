#!/usr/bin/env bash
# Usage:
#   bash predictions/cli/call_batch.sh <config_list.txt> [args forwarded to main.py]
#
# Reads one config filename per line from <config_list.txt> and invokes:
#   python predictions/cli/main.py from_file \
#       --filename predictions/configs/<config_name> \
#       "$@"
# for each entry. All args after the list file are forwarded verbatim.
# Lines that are blank or start with '#' are skipped.

set -uo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_list.txt> [args forwarded to main.py]" >&2
    exit 1
fi

config_list="$1"
shift

if [ ! -f "$config_list" ]; then
    echo "Error: config list file '$config_list' not found" >&2
    exit 1
fi

total=0
failed=0

while IFS= read -r config_name || [ -n "$config_name" ]; do
    # strip trailing \r in case the list has CRLF line endings
    config_name="${config_name%$'\r'}"
    # skip blank lines and comments
    [ -z "$config_name" ] && continue
    case "$config_name" in \#*) continue ;; esac

    total=$((total + 1))
    echo
    echo "=========================================================="
    echo "[$total] Running: $config_name"
    echo "=========================================================="

    python predictions/cli/main.py from_file \
        --filename "predictions/configs/$config_name" \
        "$@"

    status=$?
    if [ "$status" -ne 0 ]; then
        failed=$((failed + 1))
        echo ">>> FAILED ($status): $config_name" >&2
    fi
done < "$config_list"

echo
echo "=========================================================="
echo "Done. $((total - failed))/$total succeeded, $failed failed."
echo "=========================================================="
[ "$failed" -eq 0 ]