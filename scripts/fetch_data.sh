#!/usr/bin/env bash
# Downloads the USC-SIPI image database volumes used by this project and
# converts them to PNG under data/input/<volume>/.
#
#   ./scripts/fetch_data.sh                 # misc + textures + aerials
#   ./scripts/fetch_data.sh misc            # only the misc volume
#
# Requires: curl, tar, python3 with numpy and Pillow (pip install -r
# requirements.txt).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOLUMES=("$@")
if [ ${#VOLUMES[@]} -eq 0 ]; then VOLUMES=(misc textures aerials); fi

DOWNLOAD_DIR="${ROOT}/data/raw"
mkdir -p "${DOWNLOAD_DIR}"

for volume in "${VOLUMES[@]}"; do
  archive="${DOWNLOAD_DIR}/${volume}.tar.gz"
  if [ ! -f "${archive}" ]; then
    echo "Downloading SIPI volume '${volume}'..."
    curl -fL --retry 3 -o "${archive}" \
      "https://sipi.usc.edu/database/${volume}.tar.gz"
  fi
  extract_dir="${DOWNLOAD_DIR}/${volume}"
  rm -rf "${extract_dir}" && mkdir -p "${extract_dir}"
  tar -xzf "${archive}" -C "${extract_dir}"
  python3 "${ROOT}/scripts/convert_images.py" \
    --src "${extract_dir}" --dst "${ROOT}/data/input/${volume}"
done

# One large synthetic image (8192 x 8192 = 67 MP) to test scaling.
if [ ! -f "${ROOT}/data/input/synthetic/synthetic_8192x8192.png" ]; then
  python3 "${ROOT}/scripts/convert_images.py" --synthetic 8192 \
    --dst "${ROOT}/data/input/synthetic"
fi
echo "Data ready under ${ROOT}/data/input"
