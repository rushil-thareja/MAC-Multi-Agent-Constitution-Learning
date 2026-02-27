# MAC — Multi-Agent Constitution Learning

__version__ = "0.1.0"

# DSPy-style public API
from .example import Example, ErrorItem, ErrorReport, DefaultErrorAnalyzer, MAC
from .compiled import CompiledMAC
from .display import MACDisplay

__all__ = [
    "Example",
    "ErrorItem",
    "ErrorReport",
    "DefaultErrorAnalyzer",
    "MAC",
    "CompiledMAC",
    "MACDisplay",
]