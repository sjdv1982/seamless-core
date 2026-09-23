#!/bin/bash
# Run each test file in a separate pytest process. This isolates collection and
# process-global state, closes Seamless explicitly after each file, and lets
# the rest of the suite run when one file fails.
#
# Requires the `seamless1` environment (`conda activate seamless1`).
# Usage:
#   ./run-tests.sh                 # run every test file
#   ./run-tests.sh -k expression   # pass options through to pytest
#   TEST_TIMEOUT=300 ./run-tests.sh  # set a per-file timeout in seconds

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"

status=0
for test_file in tests/test_*.py; do
    echo "$test_file"
    if [ -n "${TEST_TIMEOUT:-}" ]; then
        timeout --foreground "${TEST_TIMEOUT}" \
            python -m pytest -s -p seamless_test_shutdown "$test_file" "$@" || status=1
    else
        python -m pytest -s -p seamless_test_shutdown "$test_file" "$@" || status=1
    fi
    echo "DONE $test_file"
done

exit "$status"
