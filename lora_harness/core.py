from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Union


@dataclass
class LoRAHandle:
    """
    Lightweight description of a LoRA adapter.
    """

    identifier: str
    path: Optional[str] = None
    embedding: Optional[Sequence[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExampleContext:
    """
    Benchmark-agnostic snapshot of one example.
    """

    dataset_name: str
    test_idx: Any
    messages: List[Dict[str, Any]]
    lang: str
    evaluate_fn: Callable[[str], Dict[str, Any]]
    user_prompt: Any = None
    info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset: Any = None
    chat_id: Optional[str] = None

    def evaluate(self, response: str) -> Dict[str, Any]:
        return self.evaluate_fn(response)


@dataclass
class ExampleResult:
    example: ExampleContext
    response: str
    metrics: Dict[str, Any]
    retrieved_lora: Optional[LoRAHandle]
    updated_lora: Optional[LoRAHandle]


class InMemoryLoRAStore:
    """
    Minimal registry tracking LoRA usage per example/chat with optional
    embedding-based retrieval.
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        self._last_by_chat: Dict[str, LoRAHandle] = {}

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        denom = norm_a * norm_b
        return dot_product / denom if denom else 0.0

    def record(
        self,
        *,
        example: ExampleContext,
        handle: Optional[LoRAHandle],
        metrics: Optional[Dict[str, Any]] = None,
        response: Optional[str] = None,
    ) -> None:
        record = {
            "dataset": example.dataset_name,
            "test_idx": example.test_idx,
            "chat_id": example.chat_id,
            "lora": handle,
            "metrics": metrics or {},
            "response": response,
            "embedding": None,
        }
        if handle is not None:
            record["embedding"] = handle.embedding
        self._records.append(record)
        if example.chat_id and handle is not None:
            self._last_by_chat[example.chat_id] = handle

    def get_last(self, chat_id: Optional[str]) -> Optional[LoRAHandle]:
        if chat_id is None:
            return None
        return self._last_by_chat.get(chat_id)

    def search_by_embedding(
        self, query_embedding: Sequence[float], top_k: int = 1
    ) -> List[LoRAHandle]:
        scored: List[tuple[float, LoRAHandle]] = []
        for record in self._records:
            embedding = record.get("embedding")
            handle = record.get("lora")
            if embedding is None or handle is None:
                continue
            if len(embedding) != len(query_embedding):
                continue
            score = self._cosine_similarity(query_embedding, embedding)
            scored.append((score, handle))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [handle for _, handle in scored[:top_k]]

    @property
    def records(self) -> List[Dict[str, Any]]:
        return list(self._records)


ExampleSource = Union[
    Iterable[ExampleContext],
    Callable[[], Iterator[ExampleContext]],
]


class LoRAHarness:
    """
    General per-example loop that works with any benchmark supplying an
    ``ExampleContext`` stream and a dataset-specific evaluation callable.
    """

    def __init__(
        self,
        *,
        example_source: ExampleSource,
        retrieve_lora_fn: Callable[[ExampleContext, "LoRAHarness"], Optional[LoRAHandle]],
        inference_fn: Callable[
            [ExampleContext, Optional[LoRAHandle], "LoRAHarness"], str
        ],
        finetune_fn: Optional[
            Callable[
                [ExampleContext, Optional[LoRAHandle], str, Dict[str, Any], "LoRAHarness"],
                Optional[LoRAHandle],
            ]
        ] = None,
        store: Optional[InMemoryLoRAStore] = None,
    ):
        if callable(example_source):
            self._example_source = example_source  # type: ignore[assignment]
        else:
            self._example_source = lambda: iter(example_source)
        self.retrieve_lora_fn = retrieve_lora_fn
        self.inference_fn = inference_fn
        self.finetune_fn = finetune_fn
        self.store = store or InMemoryLoRAStore()
        self.results: List[ExampleResult] = []

    def _iter_examples(self) -> Iterator[ExampleContext]:
        source = self._example_source
        if callable(source):
            return source()  # type: ignore[misc]
        return iter(source)  # pragma: no cover

    def run(self) -> Iterator[ExampleResult]:
        for example in self._iter_examples():
            retrieved = self.retrieve_lora_fn(example, self)
            response = self.inference_fn(example, retrieved, self)
            metrics = example.evaluate(response)
            updated = None
            if self.finetune_fn is not None:
                updated = self.finetune_fn(example, retrieved, response, metrics, self)
            result = ExampleResult(
                example=example,
                response=response,
                metrics=metrics,
                retrieved_lora=retrieved,
                updated_lora=updated,
            )
            self.results.append(result)
            used_handle = updated or retrieved
            self.store.record(example=example, handle=used_handle, metrics=metrics, response=response)
            yield result
