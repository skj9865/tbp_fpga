#!/usr/bin/env bash
# Run capture_and_infer.py from an SSH session with the preview window on the
# machine's physical monitor.
#
# The OAK preview needs a real X display; an SSH session has DISPLAY unset, so
# cv2 fails to open a window. The Z8's physical session is on :1 (not :0 —
# check with `ls /tmp/.X11-unix`, socket X1). Keys ('c'/'q'/'i'/'d') are read
# from BOTH the preview window and this terminal, so the whole session is
# drivable over SSH.
#
# Activate the env first (this wrapper deliberately does not use `conda run`,
# which would not give the script a tty and would break terminal key input):
#   conda activate tbp_fpga
#   scripts/ci.sh --object numenta_mug --extended-disparity
set -euo pipefail

export DISPLAY="${DISPLAY:-:1}"

if ! python -c "import depthai" 2>/dev/null; then
    echo "depthai not importable — did you 'conda activate tbp_fpga'?" >&2
    exit 1
fi

exec python "$(dirname "$(readlink -f "$0")")/capture_and_infer.py" "$@"
