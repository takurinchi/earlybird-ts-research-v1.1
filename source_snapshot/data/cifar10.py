"""CIFAR-10 input pipeline, with a fixed-seed 10% train/validation holdout."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def create_loaders(root="data", batch_size=256, seed=1, num_workers=0):
    # Content-pinned, byte-identical mirror of the official CIFAR-10 archive.
    datasets.CIFAR10.url = (
        "https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/"
        "resolve/c6e562a/cifar-10-python.tar.gz")
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([transforms.Pad(4), transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    training = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
    validation_base = datasets.CIFAR10(root, train=True, download=True, transform=eval_transform)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(training), generator=generator).tolist()
    n_validation = 5000
    validation_indices, train_indices = permutation[:n_validation], permutation[n_validation:]
    test = datasets.CIFAR10(root, train=False, download=True, transform=eval_transform)
    kwargs = {"batch_size": batch_size, "num_workers": num_workers,
              "pin_memory": torch.cuda.is_available()}
    return {
        "train": DataLoader(Subset(training, train_indices), shuffle=True, **kwargs),
        "validation": DataLoader(Subset(validation_base, validation_indices), shuffle=False, **kwargs),
        "test": DataLoader(test, shuffle=False, **kwargs),
    }
