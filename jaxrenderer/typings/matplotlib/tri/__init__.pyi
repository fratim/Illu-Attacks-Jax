"""
This type stub file was generated by pyright.
"""

from ._triangulation import Triangulation
from ._tricontour import TriContourSet, tricontour, tricontourf
from ._trifinder import TrapezoidMapTriFinder, TriFinder
from ._triinterpolate import (
    CubicTriInterpolator,
    LinearTriInterpolator,
    TriInterpolator,
)
from ._tripcolor import tripcolor
from ._triplot import triplot
from ._trirefine import TriRefiner, UniformTriRefiner
from ._tritools import TriAnalyzer

"""
Unstructured triangular grid functions.
"""
__all__ = [
    "Triangulation",
    "TriContourSet",
    "tricontour",
    "tricontourf",
    "TriFinder",
    "TrapezoidMapTriFinder",
    "TriInterpolator",
    "LinearTriInterpolator",
    "CubicTriInterpolator",
    "tripcolor",
    "triplot",
    "TriRefiner",
    "UniformTriRefiner",
    "TriAnalyzer",
]
