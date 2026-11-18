"""Early-Bird detector matching the official five-mask rolling comparison."""
from collections import deque
from dataclasses import dataclass, field
from .mask_distance import mask_distance


@dataclass
class StabilityDetector:
    threshold: float = 0.1
    mask_window: int = 5
    tm: int | None = None
    _masks: deque = field(init=False, repr=False)
    last_distances: list = field(default_factory=list)

    def __post_init__(self):
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if self.mask_window < 2:
            raise ValueError("mask_window must be at least 2")
        self._masks = deque(maxlen=self.mask_window)

    def update(self, epoch: int, mask) -> bool:
        """Add this epoch's binary mask and check it against prior masks.

        The official implementation keeps five masks, so each check compares
        the newest mask with the preceding four masks (not five distances).
        The reference code rejects distances greater than the threshold, so
        equality is accepted (distance <= threshold).
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        previous = list(self._masks)[-(self.mask_window - 1):]
        self.last_distances = [mask_distance(old, mask) for old in previous]
        self._masks.append(mask)
        if self.tm is None and len(self._masks) == self.mask_window and all(
                distance <= self.threshold for distance in self.last_distances):
            self.tm = epoch
            return True
        return False

    @property
    def detected(self) -> bool:
        return self.tm is not None
