"""Classification metrics, with macro-F1 defined for all observed/known labels."""
from sklearn.metrics import accuracy_score, f1_score
import torch


def classification_metrics(targets, predictions, labels=None):
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, labels=labels,
                                    average="macro", zero_division=0)),
    }


@torch.inference_mode()
def evaluate_classifier(model, loader, device, labels=None):
    model.eval()
    targets, predictions = [], []
    loss_sum = count = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for inputs, target, *_ in loader:
        inputs, target = inputs.to(device), target.to(device)
        logits = model(inputs)
        loss_sum += criterion(logits, target).item()
        count += target.numel()
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(target.cpu().tolist())
    if count == 0:
        raise ValueError("Evaluation loader produced no samples")
    return {"loss": loss_sum / count, **classification_metrics(targets, predictions, labels)}
