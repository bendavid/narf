export NARF_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export PYTHONPATH="${NARF_BASE}:$PYTHONPATH"
export EXTRA_CLING_ARGS="-O3"
export XRD_PARALLELEVTLOOP="16"

#workaround for bug in tf lite
# export TF_NUM_INTEROP_THREADS="1"

#workaround for lock contention issue in tflite
export TF_ENABLE_ONEDNN_OPTS="0"
export TF_DISABLE_MKL="1"

echo "Created environment variable NARF_BASE=${NARF_BASE}"
