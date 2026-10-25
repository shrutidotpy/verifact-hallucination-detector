#!/usr/bin/env bash
# One-time setup for macOS / Linux.  Run:  bash setup.sh
# To force a specific Python:            PYTHON=/path/to/python3 bash setup.sh
set -e

# PyTorch only publishes packages for certain Python versions, so pick the
# first installed Python between 3.10 and 3.13 (a too-new Python such as 3.15
# fails with "No matching distribution found for torch").
is_supported() {
  "$1" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null
}

PY="$PYTHON"
if [ -z "$PY" ]; then
  for candidate in python3.12 python3.13 python3.11 python3.10 /opt/anaconda3/bin/python3 /opt/homebrew/bin/python3 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && is_supported "$candidate"; then
      PY="$candidate"
      break
    fi
  done
fi
if [ -z "$PY" ] || ! is_supported "$PY"; then
  echo "ERROR: need Python 3.10-3.13. Install Python 3.12 from https://www.python.org/downloads/"
  exit 1
fi
echo "Using $(command -v "$PY") ($("$PY" --version))"

# Remove a half-built environment from an earlier failed attempt.
rm -rf .venv
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
echo
echo "Setup complete. Try:  .venv/bin/python -m halludetect.step1_claims"
echo "Web app:              .venv/bin/streamlit run app.py"
