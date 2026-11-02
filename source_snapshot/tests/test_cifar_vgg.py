import torch
from models.cifar_vgg import create_vgg16_bn, structurally_prune_vgg16


def test_compaction_preserves_predictions_when_all_channels_are_kept():
    torch.manual_seed(4)
    model = create_vgg16_bn().eval()
    masks = {name: torch.ones(module.num_features, dtype=torch.bool)
             for name, module in model.named_modules()
             if isinstance(module, torch.nn.BatchNorm2d)}
    compact = structurally_prune_vgg16(model, masks).eval()
    sample = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        expected, actual = model(sample), compact(sample)
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
