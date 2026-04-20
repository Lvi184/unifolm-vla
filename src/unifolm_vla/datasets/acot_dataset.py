from __future__ import annotations

from typing import Iterable, List, Iterator, Dict, Any
import torch
from torch.utils.data import IterableDataset

from .acot_reader import AgiBotWorldTaskReader
from .acot_adapter import ACOTToUnifoLMAdapter


class ACOTIterableDataset(IterableDataset):
    """
    Stream ACoT/AgiBotWorld raw data directly into UnifoLM-style samples.
    """

    def __init__(
        self,
        task_roots: List[str],
        adapter: ACOTToUnifoLMAdapter,
    ):
        super().__init__()
        self.task_roots = task_roots
        self.adapter = adapter

    def _iter_single_task(self, task_root: str) -> Iterator[Dict[str, Any]]:
        reader = AgiBotWorldTaskReader(task_root)
        for sample in reader.iter_all_steps():
            yield self.adapter(sample)

    def __iter__(self):
        for task_root in self.task_roots:
            yield from self._iter_single_task(task_root)


class TransformedIterableDataset(IterableDataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        # Estimate length: ~100 episodes per task × ~400 steps per episode = 40,000 steps per task
        self._estimated_len = len(base_dataset.task_roots) * 100 * 400

    def __iter__(self):
        for sample in self.base_dataset:
            out = self.transform(sample)
            if out is not None:
                yield out

    def __len__(self):
        return self._estimated_len
