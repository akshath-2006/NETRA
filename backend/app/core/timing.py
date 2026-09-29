"""Small timing helpers shared across the pipeline."""

from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """Rolling frames-per-second measurement over a fixed window.

    A rolling window rather than a running total, because what we care about
    on the HUD is "how fast is it going *now*", not the average since start.
    """

    def __init__(self, window: int = 45) -> None:
        self._stamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._stamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.fps:.1f}"
