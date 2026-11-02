"""Parameter count, zero sparsity, and relative reduction."""
def parameter_count(model):
    total = zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p.detach() == 0).sum().item()
    return {"parameter_count": int(total),
            "zero_parameter_count": int(zeros),
            "sparsity": zeros / total if total else 0.0}


def reduction_rate(baseline, value):
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return (baseline - value) / baseline * 100.0
