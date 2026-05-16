SCRIPT_FILE_REL_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_FILE_REL_PATH" == "" ]]; then
  SCRIPT_FILE_REL_PATH="${(%):-%N}"
fi
export NARF_BASE=$( cd "$( dirname "${SCRIPT_FILE_REL_PATH}" )" && pwd )
export PYTHONPATH="${NARF_BASE}:$PYTHONPATH"
export EXTRA_CLING_ARGS="-O3"
export XRD_PARALLELEVTLOOP="16"

# prevent Eigen from spawning spurious threads
export OMP_NUM_THREADS="1"

#workaround for lock contention issue in tflite
export TF_ENABLE_ONEDNN_OPTS="0"
export TF_DISABLE_MKL="1"

# openblas doesn't scale well beyond 64 threads currently (v0.3.29)
# so set the number of threads to min(64, ncpus)
# note that --all is needed because otherwise nproc keys on
# OMP_NUM_THREADS set above and would return 1
export OPENBLAS_NUM_THREADS=$((`nproc --all`>64 ? 64 : `nproc --all`))

echo "Created environment variable NARF_BASE=${NARF_BASE}"
