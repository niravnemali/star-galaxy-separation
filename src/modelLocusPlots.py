"""Backward-compatible shim for older notebooks.

Some notebook versions import ``modelLocusPlots as pltL`` while the current
repo uses ``plotting.py``. Re-export everything from ``plotting`` so both
import styles work without modifying the scientific workflow.
"""

from plotting import *  # noqa: F401,F403
