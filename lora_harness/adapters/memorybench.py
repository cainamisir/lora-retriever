from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MEMORYBENCH_ROOT = os.path.join(PROJECT_ROOT, "MemoryBench")
if MEMORYBENCH_ROOT not in sys.path:
    sys.path.append(MEMORYBENCH_ROOT)

from memorybench import load_memory_bench  # type: ignore
from src.dataset.base import BaseDataset  # type: ignore

from ..core import ExampleContext


class MemoryBenchExampleSource:
    """
    Turns MemoryBench datasets/domains/tasks into an ``ExampleContext`` stream
    consumable by the general-purpose :class:`LoRAHarness`.
    """

    def __init__(
        self,
        *,
        dataset_type: str,
        name: str,
        split: str = "test",
        limit_per_dataset: Optional[int] = None,
        eval_mode: bool = True,
        chat_id_getter: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], Optional[str]]
        ] = None,
    ):
        self.dataset_type = dataset_type
        self.name = name
        self.split = split
        self.limit_per_dataset = limit_per_dataset
        self.chat_id_getter = chat_id_getter or self._default_chat_id_getter
        dataset_obj = load_memory_bench(dataset_type, name, eval_mode=eval_mode)
        if dataset_type == "single":
            self._datasets: List[BaseDataset] = [dataset_obj]
        else:
            self._datasets = list(dataset_obj)

    def __iter__(self) -> Iterator[ExampleContext]:
        for dataset in self._datasets:
            split_data = dataset.dataset[self.split].to_list()
            for idx, raw_data in enumerate(split_data):
                if self.limit_per_dataset is not None and idx >= self.limit_per_dataset:
                    break
                yield self._build_context(dataset, raw_data)

    def _build_context(self, dataset: BaseDataset, raw: Dict[str, Any]) -> ExampleContext:
        data = dict(raw)
        user_prompt = data.get("input_prompt") or data.get("input_chat_messages")
        if user_prompt is None:
            raise ValueError(
                f"Example {data.get('test_idx')} in {dataset.dataset_name} "
                "has neither 'input_prompt' nor 'input_chat_messages'."
            )
        messages = dataset.get_initial_chat_messages(data["test_idx"])
        info = data["info"]

        def _evaluate_fn(
            response: str,
            *,
            dataset=dataset,
            user_prompt=user_prompt,
            info=info,
        ) -> Dict[str, Any]:
            return dataset.evaluate_single(user_prompt, info, response)

        chat_id = self.chat_id_getter(data, info)
        metadata = {
            "raw": data,
            "split": self.split,
            "dataset_type": self.dataset_type,
            "set_name": self.name,
        }
        return ExampleContext(
            dataset_name=dataset.dataset_name,
            test_idx=int(data["test_idx"]),
            messages=messages,
            lang=data.get("lang", "en"),
            evaluate_fn=_evaluate_fn,
            user_prompt=user_prompt,
            info=info,
            metadata=metadata,
            dataset=dataset,
            chat_id=chat_id,
        )

    @staticmethod
    def _default_chat_id_getter(
        raw: Dict[str, Any], info: Dict[str, Any]
    ) -> Optional[str]:
        for key in ("dialog_id", "conversation_id", "chat_id", "session_id"):
            value = raw.get(key)
            if value is not None:
                return str(value)
        for key in ("dialog_id", "conversation_id", "chat_id", "session_id"):
            value = info.get(key)
            if value is not None:
                return str(value)
        return None
