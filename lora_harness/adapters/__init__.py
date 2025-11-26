"""
Adapters that turn specific benchmarks into ``ExampleContext`` streams.
"""

from .memorybench import MemoryBenchExampleSource

__all__ = ["MemoryBenchExampleSource"]
