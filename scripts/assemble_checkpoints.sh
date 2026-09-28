#!/usr/bin/env bash
# Reassemble split HDiT checkpoints downloaded from a GitHub Release.
# Run this in the directory that contains the .part-* files and SHA256SUMS.
set -euo pipefail

FFHQ_NAME="hdit-ffhq256-unconditional.pt"
IN_NAME="hdit-imagenet256-unconditional.pt"

assemble() {
  local name="$1"
  local parts=( "${name}.part-"* )
  if [[ ! -e "${parts[0]}" ]]; then
    echo "skip ${name}: no parts found" >&2
    return 0
  fi
  echo "Assembling ${name} from ${#parts[@]} part(s)..."
  cat "${name}.part-"* > "${name}"
  ls -lh "${name}"
}

assemble "$FFHQ_NAME"
assemble "$IN_NAME"

if [[ -f SHA256SUMS ]]; then
  echo "Verifying checksums..."
  sha256sum -c SHA256SUMS
else
  echo "warning: SHA256SUMS not found; skipped verification" >&2
fi

echo "Done. Point configs/model/hdit_*.yaml path keys at the .pt files in this directory."
