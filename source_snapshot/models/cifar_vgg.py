"""CIFAR VGG16-BN matching the official Early-Bird CIFAR VGG feature layout."""
import copy
import torch
from torch import nn


VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
             512, 512, 512, "M", 512, 512, 512]


class CifarVGG16BN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        layers, in_channels = [], 3
        for channels in VGG16_CFG:
            if channels == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                               nn.BatchNorm2d(channels), nn.ReLU(inplace=True)])
                in_channels = channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Linear(in_channels, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, (2.0 / (module.kernel_size[0] ** 2 * module.out_channels)) ** 0.5)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 0.5)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


def create_vgg16_bn(num_classes=10):
    return CifarVGG16BN(num_classes=num_classes)


def structurally_prune_vgg16(model, masks, return_parameter_slices=False):
    """Return a narrower VGG, optionally with per-parameter slicing metadata.

    The slice metadata maps each original named parameter to the dimensions and
    retained indices used in the compact parameter. Optimizer state tensors with
    the original parameter shape can use the same mapping.
    """
    layers = []
    parameter_slices = {}
    channel_indices = {}
    input_idx = torch.arange(3)
    module_index = 0
    features = list(model.features.children())
    while module_index < len(features):
        layer = features[module_index]
        if isinstance(layer, nn.Conv2d):
            bn_name = f"features.{module_index + 1}"
            if bn_name not in masks:
                raise KeyError(f"Missing channel mask for {bn_name}")
            output_idx = torch.where(masks[bn_name].bool())[0].to(layer.weight.device)
            if output_idx.numel() == 0:
                raise ValueError(f"Mask for {bn_name} removes every channel")
            in_idx = input_idx.to(layer.weight.device)
            channel_indices[bn_name] = output_idx.detach().cpu()
            conv_name = f"features.{module_index}"
            parameter_slices[f"{conv_name}.weight"] = ((0, output_idx), (1, in_idx))
            if layer.bias is not None:
                parameter_slices[f"{conv_name}.bias"] = ((0, output_idx),)
            if features[module_index + 1].affine:
                parameter_slices[f"{bn_name}.weight"] = ((0, output_idx),)
                parameter_slices[f"{bn_name}.bias"] = ((0, output_idx),)
            conv = nn.Conv2d(in_idx.numel(), output_idx.numel(), layer.kernel_size,
                             layer.stride, layer.padding, layer.dilation,
                             layer.groups, bias=layer.bias is not None,
                             padding_mode=layer.padding_mode).to(layer.weight.device, layer.weight.dtype)
            with torch.no_grad():
                conv.weight.copy_(layer.weight.index_select(0, output_idx).index_select(1, in_idx))
                if layer.bias is not None:
                    conv.bias.copy_(layer.bias.index_select(0, output_idx))
            old_bn = features[module_index + 1]
            bn = nn.BatchNorm2d(output_idx.numel(), old_bn.eps, old_bn.momentum,
                                old_bn.affine, old_bn.track_running_stats).to(layer.weight.device)
            keep_cpu = output_idx.cpu()
            with torch.no_grad():
                if old_bn.affine:
                    bn.weight.copy_(old_bn.weight.index_select(0, output_idx))
                    bn.bias.copy_(old_bn.bias.index_select(0, output_idx))
                if old_bn.track_running_stats:
                    bn.running_mean.copy_(old_bn.running_mean.index_select(0, output_idx))
                    bn.running_var.copy_(old_bn.running_var.index_select(0, output_idx))
                    bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
            layers.extend([conv, bn, copy.deepcopy(features[module_index + 2])])
            input_idx = keep_cpu
            module_index += 3
        else:
            layers.append(copy.deepcopy(layer))
            module_index += 1
    compact = CifarVGG16BN.__new__(CifarVGG16BN)
    nn.Module.__init__(compact)
    compact.features = nn.Sequential(*layers)
    compact.avgpool = copy.deepcopy(model.avgpool)
    last_idx = input_idx.to(model.classifier.weight.device)
    channel_indices["classifier.input"] = last_idx.detach().cpu()
    parameter_slices["classifier.weight"] = ((1, last_idx),)
    if model.classifier.bias is not None:
        parameter_slices["classifier.bias"] = ()
    linear = nn.Linear(last_idx.numel(), model.classifier.out_features,
                       bias=model.classifier.bias is not None).to(
                           model.classifier.weight.device, model.classifier.weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(model.classifier.weight.index_select(1, last_idx))
        if model.classifier.bias is not None:
            linear.bias.copy_(model.classifier.bias)
    compact.classifier = linear
    if return_parameter_slices:
        return compact, parameter_slices, channel_indices
    return compact
