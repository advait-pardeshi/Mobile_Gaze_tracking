#!/usr/bin/env bash
# Build and run the calibration replay harness.
#
# Compiles the SHIPPING CalibrationModel.swift and ScreenMapper.swift together
# with the harness, so the A/B scores the real code rather than a port of it.
# Any drift between them becomes a compile error here rather than a wrong
# number.
#
#   ./Tools/replay_calibration.sh ~/Downloads --sweep --per-cell
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="${TMPDIR:-/tmp}/calibration-replay"

swiftc -O \
  "$root/Sources/CalibrationModel.swift" \
  "$root/Sources/ScreenMapper.swift" \
  "$root/Tools/CalibrationReplay/main.swift" \
  -o "$bin"

exec "$bin" "$@"
