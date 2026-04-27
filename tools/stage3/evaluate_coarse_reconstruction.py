"""Deprecated wrapper for the controlled benchmark entry point.

Use the standard module path instead:
    .venv/bin/python -m tools.stage3.controlled.evaluate_coarse_reconstruction
"""

from tools.stage3.controlled.evaluate_coarse_reconstruction import main


if __name__ == "__main__":
    main()
