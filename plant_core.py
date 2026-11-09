"""Compatibility import; new code should use :mod:`industrial_reliability_hub.core.plant_core`."""
import sys
from industrial_reliability_hub.core import plant_core as _implementation

sys.modules[__name__] = _implementation
