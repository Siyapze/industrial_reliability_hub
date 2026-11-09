"""Compatibility import; new code should use :mod:`industrial_reliability_hub.serialization`."""
import sys
from industrial_reliability_hub import serialization as _implementation

sys.modules[__name__] = _implementation
