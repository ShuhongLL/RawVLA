#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ASSETS_ROOT="${REPO_ROOT}/third_party/RoboTwin/assets"
ASSETS_SOURCE="${ROBOTWIN_ASSETS_SOURCE:-}"
ASSET_GROUPS=(background_texture embodiments objects)

if [[ -n "${ASSETS_SOURCE}" ]]; then
  ASSETS_SOURCE="$(cd -- "${ASSETS_SOURCE}" && pwd)"
  for group in "${ASSET_GROUPS[@]}"; do
    source_path="${ASSETS_SOURCE}/${group}"
    target_path="${ASSETS_ROOT}/${group}"
    if [[ ! -d "${source_path}" ]]; then
      echo "missing RoboTwin asset directory: ${source_path}" >&2
      exit 1
    fi
    if [[ -e "${target_path}" || -L "${target_path}" ]]; then
      echo "reuse ${target_path}"
    else
      ln -s "${source_path}" "${target_path}"
      echo "linked ${target_path} -> ${source_path}"
    fi
  done
else
  command -v unzip >/dev/null || {
    echo "unzip is required to extract RoboTwin assets" >&2
    exit 1
  }
  (
    cd -- "${ASSETS_ROOT}"
    python _download.py
    for group in "${ASSET_GROUPS[@]}"; do
      archive="${group}.zip"
      if [[ ! -f "${archive}" ]]; then
        echo "official asset download did not produce ${archive}" >&2
        exit 1
      fi
      unzip -q -o "${archive}"
    done
  )
fi

for required in \
  "${ASSETS_ROOT}/objects/objaverse/list.json" \
  "${ASSETS_ROOT}/embodiments/aloha-agilex"; do
  if [[ ! -e "${required}" ]]; then
    echo "RoboTwin asset validation failed: ${required}" >&2
    exit 1
  fi
done

echo "RoboTwin assets ready: ${ASSETS_ROOT}"
