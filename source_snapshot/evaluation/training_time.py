"""Wall-clock timing helpers."""
import time


class Timer:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.seconds = time.perf_counter() - self.started


def measure_training(train_callable, detection_callable=None):
    with Timer() as total:
        with Timer() as training:
            train_callable()
        detection = 0.0
        if detection_callable is not None:
            with Timer() as detection_timer:
                detection_callable()
            detection = detection_timer.seconds
    return {"training_seconds": training.seconds,
            "mask_detection_seconds": detection,
            "total_seconds": total.seconds}
