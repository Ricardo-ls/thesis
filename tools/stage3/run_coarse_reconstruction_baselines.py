"""Deprecated wrapper for the controlled benchmark entry point.

Use the standard module path instead:
    .venv/bin/python -m tools.stage3.controlled.run_coarse_reconstruction_baselines
"""

from tools.stage3.controlled.run_coarse_reconstruction_baselines import main


if __name__ == "__main__":
    main()
