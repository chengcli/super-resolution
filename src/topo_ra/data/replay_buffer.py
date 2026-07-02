from __future__ import annotations

import random
from collections import deque
from typing import Deque


class ReplayBuffer:
    def __init__(self, capacity: int = 128) -> None:
        self.capacity = int(capacity)
        self._items: Deque[dict] = deque(maxlen=self.capacity)

    def add(self, item: dict) -> None:
        self._items.append(item)

    def extend(self, items: list[dict]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> list[dict]:
        if not self._items:
            raise ValueError("Cannot sample from an empty replay buffer")
        count = min(int(batch_size), len(self._items))
        return random.sample(list(self._items), count)

    def __len__(self) -> int:
        return len(self._items)
