"""Optional FLOPs counter adapter (requires user-selected compatible backend)."""
def count_flops(model, example_inputs, counter=None):
    if counter is None:
        return None
    # counter(model, example_inputs) must return a numeric operation count.
    value = counter(model, example_inputs)
    return int(value)


def count_dense_flops(model, input_shape=(1, 3, 32, 32), device="cpu"):
    """Count multiply/add FLOPs for Conv2d and Linear layers with hooks."""
    import torch
    total = 0
    handles = []
    def conv_hook(module, _inputs, output):
        nonlocal total
        batch, channels, height, width = output.shape
        kh, kw = module.kernel_size
        total += int(2 * batch * channels * height * width *
                     (module.in_channels // module.groups) * kh * kw)
    def linear_hook(module, inputs, _output):
        nonlocal total
        total += int(2 * inputs[0].shape[0] * module.in_features * module.out_features)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(input_shape, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return total
