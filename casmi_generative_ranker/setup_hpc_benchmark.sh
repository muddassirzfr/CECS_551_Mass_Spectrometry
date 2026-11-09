#!/bin/bash
# Run on the login host after staging code and prepared data. Existing envs stay intact.
set -euo pipefail
task_root="${CASMI_RUN_DIR:-$HOME/casmi-benchmark-20261107}"
runtime="${CASMI_PYTHON:-$HOME/vizwiz-venv/bin/python}"
mkdir -p "$task_root"
task_dependencies="$(mktemp -d "$task_root/deps-install.XXXXXX")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# A fresh target avoids changing modules while another Python process imports them.
"$runtime" -m pip install --target "$task_dependencies" \
  'pandas>=2.1,<3' 'pyarrow>=15' 'rdkit==2026.3.3' \
  'scikit-learn>=1.4,<1.8' 'lightgbm>=4.3' joblib threadpoolctl \
  'scipy>=1.15' python-dateutil pytz tzdata six Pillow narwhals
PYTHONPATH="$task_dependencies${PYTHONPATH:+:$PYTHONPATH}" "$runtime" -c \
  'import torch,pandas,pyarrow,rdkit,lightgbm,sklearn,tokenizers; print("Dependencies ready",torch.__version__,rdkit.__version__)'
if [ -e "$task_root/dependencies_current" ] && [ ! -L "$task_root/dependencies_current" ]; then
  echo 'dependencies_current must be a symlink' >&2
  exit 1
fi
ln -sfn "$task_dependencies" "$task_root/dependencies_current"
