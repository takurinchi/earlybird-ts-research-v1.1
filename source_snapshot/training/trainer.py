"""Generic classification epoch routines; adapt loss/batch unpacking as needed."""
import torch


def unpack_batch(batch, device):
    inputs, targets = batch[:2]
    return inputs.to(device), targets.to(device)


def train_epoch(model, loader, optimizer, criterion, device, mask_manager=None):
    model.train(); total_loss = seen = correct = 0
    for batch in loader:
        x, y = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        if mask_manager:
            mask_manager.apply()
        total_loss += loss.item() * y.size(0)
        seen += y.size(0)
        correct += (logits.detach().argmax(1) == y).sum().item()
    if not seen:
        raise ValueError("Training loader produced no samples")
    return {"train_loss": total_loss / seen, "train_accuracy": correct / seen}


@torch.inference_mode()
def evaluate_loss(model, loader, criterion, device):
    model.eval(); total_loss = seen = 0
    for batch in loader:
        x, y = unpack_batch(batch, device)
        total_loss += criterion(model(x), y).item() * y.size(0)
        seen += y.size(0)
    if not seen:
        raise ValueError("Validation loader produced no samples")
    return total_loss / seen
