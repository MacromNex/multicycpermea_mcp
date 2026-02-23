"""
Shared library for MCP cyclic peptide scripts.

This library contains common functions extracted and simplified from the original
repository code to minimize dependencies while maintaining functionality.
"""

__version__ = "1.0.0"
__author__ = "MCP CycPep Extraction Pipeline"

from . import molecules
from . import io
from . import validation

__all__ = [
    'molecules',
    'io',
    'validation'
]