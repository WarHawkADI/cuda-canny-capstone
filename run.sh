#!/usr/bin/env bash
# End-to-end demo: build, fetch data, run the GPU pipeline in several
# configurations and collect proof-of-execution artifacts under results/.
#
#   ./run.sh            # full demo (~2-3 minutes on an RTX 2050 laptop GPU)
#   QUICK=1 ./run.sh    # only the misc volume (fast smoke test)
set -euo pipefail
cd "$(dirname "$0")"

BIN=./bin/cuda_canny
RESULTS=results
LOGS="${RESULTS}/logs"

make
mkdir -p "${LOGS}"

if [ ! -d data/input/misc ]; then
  if [ "${QUICK:-0}" = "1" ]; then ./scripts/fetch_data.sh misc
  else ./scripts/fetch_data.sh; fi
fi

nvidia-smi | tee "${LOGS}/00_nvidia_smi.log"
${BIN} --help > "${LOGS}/00_help.log"

echo "### 1. Stage-by-stage output + bit-exact verification (misc volume)"
${BIN} --input data/input/misc --output "${RESULTS}/misc" \
  --verify --save-stages --repeat 3 --workers 1 \
  | tee "${LOGS}/01_verify_misc.log"

echo "### 2. Histogram equalization variant"
${BIN} --input data/input/misc --output "${RESULTS}/equalize" \
  --equalize --save-stages --workers 2 \
  | tee "${LOGS}/02_equalize_misc.log"

if [ "${QUICK:-0}" != "1" ]; then
  echo "### 3. Verification + speedup on the whole dataset (incl. 8192x8192)"
  ${BIN} --input data/input --output "${RESULTS}/all_verify" \
    --verify --repeat 3 --workers 1 \
    | tee "${LOGS}/01_verify_all.log"

  echo "### 4. Batch throughput vs. number of concurrent CUDA streams"
  for w in 1 2 4 8; do
    ${BIN} --input data/input --output "/tmp/cuda_canny_scaling" \
      --workers "${w}" --quiet | tee "${LOGS}/03_throughput_w${w}.log"
  done

  echo "### 5. Figures"
  python3 scripts/make_report.py --results "${RESULTS}" --data data/input \
    | tee "${LOGS}/04_report.log"
fi
echo "Done. Artifacts are in ${RESULTS}/"
