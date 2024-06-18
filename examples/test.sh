#!/bin/sh

# Private variables
SCRIPT_DIR=$(dirname $(dirname $(realpath "$0")))/"scripts"

# Environment Setting
set -x  # print the commands

# Arguments
PY_ARGS=${@}  # Additional args

# Run the script
python ${SCRIPT_DIR}/"test.py" \
    ${PY_ARGS}
