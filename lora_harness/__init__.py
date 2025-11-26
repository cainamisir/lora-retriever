"""
General-purpose LoRA harness utilities.

This package is benchmark-agnostic. Pass in any example source that yields
``ExampleContext`` objects and the harness will take care of the per-example
loop:

1. Retrieve the adapter for the current example (e.g., last adapter used for
   the chat, or the nearest neighbor via embeddings).
2. Snap the adapter onto your model and generate a response.
3. Evaluate the response with the dataset-provided evaluator embedded in the
   context.
4. Optionally fine-tune / update the adapter, record the outcome, and yield an
   :class:`ExampleResult`.

Adapters for specific benchmarks (e.g., MemoryBench) live in
``lora_harness.adapters``.
"""

from .core import (
    ExampleContext,
    ExampleResult,
    InMemoryLoRAStore,
    LoRAHandle,
    LoRAHarness,
)

__all__ = [
    "ExampleContext",
    "ExampleResult",
    "InMemoryLoRAStore",
    "LoRAHandle",
    "LoRAHarness",
]
