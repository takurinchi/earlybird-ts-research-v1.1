"""Explicit t_s resolution and protection against online future-mask leakage."""


def select_mask(mode, current_epoch, ts_epoch, current_mask=None,
                stable_mask=None, available_mask_epoch=None):
    if mode == "online":
        if current_epoch < ts_epoch:
            return None
        if current_mask is None:
            raise ValueError("Online transition requires the mask available at t_s")
        if available_mask_epoch is None or available_mask_epoch > current_epoch:
            raise ValueError("Online mode requires a mask timestamped no later than current_epoch")
        return current_mask
    if mode == "posthoc":
        if stable_mask is None:
            raise ValueError("Posthoc comparison requires the detected stable mask")
        return stable_mask
    raise ValueError("mode must be 'online' or 'posthoc'")


def relative_ts_epochs(tm, before, near, after, fixed):
    """Resolve user-provided epoch offsets; no offsets are assumed implicitly."""
    if tm is None:
        raise ValueError("Cannot resolve relative t_s values when t_m was not detected")
    return {"before": max(1, tm - int(before)), "near": max(1, tm + int(near)),
            "after": max(1, tm + int(after)), "fixed": int(fixed)}
