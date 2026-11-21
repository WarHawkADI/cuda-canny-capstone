#!/usr/bin/env bash
# Packs the proof-of-execution artifacts produced by run.sh into
# proof_of_execution.zip (for the Coursera upload).
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f proof_of_execution.zip
python3 - <<'EOF'
import pathlib
import zipfile

root = pathlib.Path("results")
with zipfile.ZipFile("proof_of_execution.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(root.rglob("*")):
        # The equalize stage dumps are large; its CSV, log and the
        # comparison figure in report/ are enough as evidence.
        if path.is_file() and not (path.parts[1] == "equalize"
                                   and path.suffix == ".png"):
            z.write(path, pathlib.Path("proof_of_execution") / path.relative_to(root))
print("wrote proof_of_execution.zip")
EOF
ls -lh proof_of_execution.zip
