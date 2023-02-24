export NARF_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export PYTHONPATH="${NARF_BASE}:$PYTHONPATH"

# silence tensorflow warnings
# TODO check if this is still needed when tensorflow is updated to 2.11+
export TF_CPP_MIN_LOG_LEVEL=3

echo "Created environment variable NARF_BASE=${NARF_BASE}"
