"""Deprecated wrapper for the controlled benchmark entry point.

Use the standard module path instead:
    .venv/bin/python -m tools.stage3.controlled.build_controlled_degradation
"""

from tools.stage3.controlled.build_controlled_degradation import main


if __name__ == "__main__":
    main()
