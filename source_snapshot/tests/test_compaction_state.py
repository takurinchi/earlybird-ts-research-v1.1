import torch
import torch.nn.functional as F

from evaluation.flops import count_dense_flops
from evaluation.parameter_count import parameter_count
from evaluation.structured_metrics import (
    estimate_vgg16_compact_flops,
    estimate_vgg16_compact_parameters,
)
from models.cifar_vgg import create_vgg16_bn, structurally_prune_vgg16
from pruning.channel_manager import BNChannelMaskManager
from training.builders import build_optimizer, build_scheduler
from training.optimizer_state import migrate_optimizer_state, migrate_scheduler_state


def _half_masks(model):
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.affine:
            mask = torch.zeros(module.num_features, dtype=torch.bool)
            mask[::2] = True
            masks[name] = mask
    return masks


def _slice(value, selectors):
    for dim, index in selectors:
        value = value.index_select(dim, index.to(value.device))
    return value


def _training_config():
    return {
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "lr_schedule": {"milestones": [2, 4], "gamma": 0.1},
    }


def test_compacted_vgg_matches_gated_model_and_structural_estimates():
    torch.manual_seed(11)
    model = create_vgg16_bn().eval()
    masks = _half_masks(model)
    x = torch.randn(1, 3, 32, 32)
    manager = BNChannelMaskManager(model, masks)
    with torch.inference_mode():
        gated_logits = model(x)
    compact, _slices, _indices = structurally_prune_vgg16(
        model, masks, return_parameter_slices=True)
    compact.eval()
    with torch.inference_mode():
        compact_logits = compact(x)
    manager.close()

    torch.testing.assert_close(gated_logits, compact_logits, rtol=1e-4, atol=1e-5)
    assert estimate_vgg16_compact_parameters(model, masks) == parameter_count(compact)["parameter_count"]
    assert estimate_vgg16_compact_flops(model, masks) == count_dense_flops(compact)


def test_optimizer_and_scheduler_state_are_migrated_by_surviving_channels():
    torch.manual_seed(13)
    model = create_vgg16_bn()
    masks = _half_masks(model)
    config = _training_config()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 2])
    F.cross_entropy(model(x), y).backward()
    optimizer.step()
    scheduler.step()

    compact, parameter_slices, _indices = structurally_prune_vgg16(
        model, masks, return_parameter_slices=True)
    compact_optimizer = build_optimizer(compact, config)
    compact_scheduler = build_scheduler(compact_optimizer, config)
    migrate_optimizer_state(
        optimizer, compact_optimizer, model, compact, parameter_slices)
    migrate_scheduler_state(scheduler, compact_scheduler, optimizer, compact_optimizer)

    old_parameters = dict(model.named_parameters())
    compact_parameters = dict(compact.named_parameters())
    for name, old_parameter in old_parameters.items():
        if old_parameter not in optimizer.state:
            continue
        old_buffer = optimizer.state[old_parameter]["momentum_buffer"]
        new_buffer = compact_optimizer.state[compact_parameters[name]]["momentum_buffer"]
        expected = _slice(old_buffer, parameter_slices.get(name, ()))
        torch.testing.assert_close(new_buffer, expected, rtol=0, atol=0)

    assert compact_scheduler.last_epoch == scheduler.last_epoch
    assert compact_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    scheduler.step()
    compact_scheduler.step()
    assert compact_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
