#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Please install Python 3.10+ first."
  exit 1
fi

PYTHON_VERSION="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

PYTHON_OK="$(python3 - <<'PY'
import sys
print("yes" if sys.version_info >= (3, 10) else "no")
PY
)"

if [[ "$PYTHON_OK" != "yes" ]]; then
  echo "Detected Python $PYTHON_VERSION. Python 3.10 or newer is required."
  exit 1
fi

echo "Repository root: $REPO_ROOT"
echo "Using python3 version: $PYTHON_VERSION"

if [[ ! -d ".venv" ]]; then
  echo "Creating local virtual environment at .venv/"
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

mkdir -p .vscode

cat > .vscode/settings.json <<'JSON'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.analysis.extraPaths": [
    "${workspaceFolder}"
  ],
  "terminal.integrated.env.osx": {
    "PYTHONPYCACHEPREFIX": "/tmp",
    "MPLBACKEND": "Agg",
    "MPLCONFIGDIR": "/tmp/mpl"
  },
  "terminal.integrated.cwd": "${workspaceFolder}"
}
JSON

cat > .vscode/extensions.json <<'JSON'
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance"
  ]
}
JSON

cat > .vscode/tasks.json <<'JSON'
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Bootstrap Environment",
      "type": "shell",
      "command": "./bootstrap_vscode_env.sh",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": []
    },
    {
      "label": "Train Stage2 none seed42 100epoch",
      "type": "shell",
      "command": "PYTHONPYCACHEPREFIX=/tmp MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -u -m tools.prior.train.train_ddpm_eth_ucy_h128 --variant none --epochs 100 --batch_size 128 --timesteps 100 --hidden_dim 128 --random_seed 42",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": []
    }
  ]
}
JSON

echo
echo "Environment bootstrap completed."
echo
echo "Virtual environment:"
echo "  $REPO_ROOT/.venv"
echo
echo "Activate later with:"
echo "  source .venv/bin/activate"
echo
echo "Quick training example:"
echo "  PYTHONPYCACHEPREFIX=/tmp MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl ./.venv/bin/python -u -m tools.prior.train.train_ddpm_eth_ucy_h128 --variant none --epochs 100 --batch_size 128 --timesteps 100 --hidden_dim 128 --random_seed 42"
echo
echo "Opening an interactive shell with the environment activated..."
exec "$SHELL" -i
