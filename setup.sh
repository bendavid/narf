export NARF_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export PYTHONPATH="${NARF_BASE}:$PYTHONPATH"
export EXTRA_CLING_ARGS="-O3"

echo "Created environment variable NARF_BASE=${NARF_BASE}"
