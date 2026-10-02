#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${ROBOTWIN_PYTHON:-python}"

echo "Installing the necessary packages with ${PYTHON_BIN} ..."
"${PYTHON_BIN}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"

echo "Checking optional pytorch3d ..."
if [[ "${ROBOTWIN_INSTALL_PYTORCH3D:-0}" == "1" ]]; then
    MAX_JOBS="${MAX_JOBS:-1}" "${PYTHON_BIN}" -m pip install \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
else
    echo "Skipping optional pytorch3d; RGB evaluation does not use it."
fi

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$("${PYTHON_BIN}" -m pip show sapien | awk '/^Location:/{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
if [[ -f "${URDF_LOADER}" ]]; then
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${URDF_LOADER}"
fi


echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$("${PYTHON_BIN}" -m pip show mplib | awk '/^Location:/{print $2}')/mplib

# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
if [[ -f "${PLANNER}" ]]; then
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${PLANNER}"
fi

echo "Installing Curobo ..."
CUROBO_SRC="${ROBOTWIN_CUROBO_SRC:-${ROBOTWIN_ROOT}/envs/curobo}"
if [[ ! -d "${CUROBO_SRC}/.git" ]]; then
    git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "${CUROBO_SRC}"
fi
cd "${CUROBO_SRC}"
# CUDA 13 builds extensions as C++20, where std::lerp conflicts with the
# unused helper overloads in CuRobo 0.7.8. Rename them and keep nvcc at one
# worker to avoid vmap pressure on large multi-GPU hosts.
sed -i 's/"--threads=8"/"--threads=1"/' setup.py
if grep -Eq 'float[234]? lerp\(' src/curobo/curobolib/cpp/helper_math.h; then
    sed -i -E 's/(float[234]? )lerp\(/\1curobo_lerp(/g' src/curobo/curobolib/cpp/helper_math.h
fi
export MAX_JOBS="${MAX_JOBS:-1}"
export CCACHE_DISABLE="${CCACHE_DISABLE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
"${PYTHON_BIN}" -m pip install 'setuptools>=77' ninja
# CuRobo leaves scikit-image unpinned. Newer releases require SciPy >=1.11
# and silently replace RoboTwin's known-good SciPy 1.10.1.
"${PYTHON_BIN}" -m pip install 'scipy==1.10.1' 'scikit-image==0.21.0'
"${PYTHON_BIN}" -m pip install -e . --no-build-isolation
"${PYTHON_BIN}" -m pip install warp-lang==1.12.0
cd "${ROBOTWIN_ROOT}"

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See INSTALLATION.md for more instructions."
