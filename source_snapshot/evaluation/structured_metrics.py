"""Cheap shape-based structural estimates for a masked CIFAR VGG16-BN."""
import math

import torch
from torch import nn


def _pair(value):
    return value if isinstance(value, tuple) else (value, value)


def _output_size(size, kernel, stride, padding, dilation=1, ceil_mode=False):
    numerator = size + 2 * padding - dilation * (kernel - 1) - 1
    value = (math.ceil(numerator / stride) if ceil_mode else math.floor(numerator / stride)) + 1
    if ceil_mode and (value - 1) * stride >= size + padding:
        value -= 1
    return int(value)


def _pool_output(height, width, layer):
    kernel = _pair(layer.kernel_size)
    stride = _pair(layer.stride if layer.stride is not None else layer.kernel_size)
    padding = _pair(layer.padding)
    dilation = _pair(getattr(layer, "dilation", 1))
    ceil_mode = bool(getattr(layer, "ceil_mode", False))
    return (
        _output_size(height, kernel[0], stride[0], padding[0], dilation[0], ceil_mode),
        _output_size(width, kernel[1], stride[1], padding[1], dilation[1], ceil_mode),
    )


def estimate_vgg16_compact_flops(model, masks, input_shape=(1, 3, 32, 32)):
    """Estimate Conv2d/Linear FLOPs for the physically compact VGG shape.

    The counting convention matches ``count_dense_flops``: multiply and add
    count as two operations; BN, activation, pooling, and bias additions are
    excluded.
    """
    batch, _channels, height, width = input_shape
    current_channels = int(input_shape[1])
    total = 0
    for index, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            bn_name = f"features.{index + 1}"
            if bn_name not in masks:
                raise KeyError(f"Missing channel mask for {bn_name}")
            out_channels = int(masks[bn_name].sum().item())
            if out_channels < 1:
                raise ValueError(f"Mask for {bn_name} removes every channel")
            height = _output_size(height, layer.kernel_size[0], layer.stride[0],
                                  layer.padding[0], layer.dilation[0])
            width = _output_size(width, layer.kernel_size[1], layer.stride[1],
                                 layer.padding[1], layer.dilation[1])
            total += (2 * batch * out_channels * height * width *
                      (current_channels // layer.groups) *
                      layer.kernel_size[0] * layer.kernel_size[1])
            current_channels = out_channels
        elif isinstance(layer, nn.MaxPool2d):
            height, width = _pool_output(height, width, layer)

    height, width = _pool_output(height, width, model.avgpool)
    total += 2 * batch * current_channels * height * width * model.classifier.out_features
    return int(total)


def estimate_vgg16_compact_parameters(model, masks):
    """Estimate trainable parameter count after applying the channel masks."""
    total = 0
    input_channels = 3
    for index, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            bn_name = f"features.{index + 1}"
            if bn_name not in masks:
                raise KeyError(f"Missing channel mask for {bn_name}")
            output_channels = int(masks[bn_name].sum().item())
            if output_channels < 1:
                raise ValueError(f"Mask for {bn_name} removes every channel")
            total += (output_channels * (input_channels // layer.groups) *
                      layer.kernel_size[0] * layer.kernel_size[1])
            if layer.bias is not None:
                total += output_channels
            batch_norm = model.features[index + 1]
            if batch_norm.affine:
                total += 2 * output_channels
            input_channels = output_channels
    total += input_channels * model.classifier.out_features
    if model.classifier.bias is not None:
        total += model.classifier.out_features
    return int(total)
