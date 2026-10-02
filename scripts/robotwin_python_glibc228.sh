#!/usr/bin/env bash
set -euo pipefail

# Run the RoboTwin Python 3.10 environment on the CentOS 7 Volc image by
# borrowing the glibc 2.28 sysroot already installed with the FastWAM env.
ROBOTWIN_ENV_ROOT="${ROBOTWIN_ENV_ROOT:-${CONDA_PREFIX:-}}"
ROBOTWIN_GLIBC_SYSROOT="${ROBOTWIN_GLIBC_SYSROOT:-}"
ROBOTWIN_LOADER="${ROBOTWIN_GLIBC_SYSROOT}/lib64/ld-linux-x86-64.so.2"
ROBOTWIN_PYTHON_BIN="${ROBOTWIN_ENV_ROOT}/bin/python"
ROBOTWIN_TORCH_LIB_ROOT="${ROBOTWIN_TORCH_OVERLAY:-${ROBOTWIN_ENV_ROOT}/lib/python3.10/site-packages}/torch/lib"

[[ -n "${ROBOTWIN_ENV_ROOT}" ]] || { echo "Set ROBOTWIN_ENV_ROOT or activate its conda environment." >&2; exit 1; }
[[ -n "${ROBOTWIN_GLIBC_SYSROOT}" ]] || { echo "Set ROBOTWIN_GLIBC_SYSROOT to a glibc 2.28 sysroot." >&2; exit 1; }
[[ -x "${ROBOTWIN_LOADER}" ]] || { echo "Missing glibc loader: ${ROBOTWIN_LOADER}" >&2; exit 1; }
[[ -x "${ROBOTWIN_PYTHON_BIN}" ]] || { echo "Missing RoboTwin Python: ${ROBOTWIN_PYTHON_BIN}" >&2; exit 1; }

ROBOTWIN_LIBRARY_PATH="${ROBOTWIN_GLIBC_SYSROOT}/lib64:${ROBOTWIN_ENV_ROOT}/lib:${ROBOTWIN_TORCH_LIB_ROOT}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  ROBOTWIN_LIBRARY_PATH="${ROBOTWIN_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
fi

exec "${ROBOTWIN_LOADER}" --library-path "${ROBOTWIN_LIBRARY_PATH}" "${ROBOTWIN_PYTHON_BIN}" "$@"
