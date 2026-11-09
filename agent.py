"""Compatibility import; new code should use :mod:`industrial_reliability_hub.agent`."""
import sys
from industrial_reliability_hub import agent as _implementation

sys.modules[__name__] = _implementation
