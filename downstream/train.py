import argparse
from pathlib import Path
from downstream.model import CoolProjectionHead

import torch
from torch import nn
from torch.utils.data import DataLoader

from downstream.datasets.StarSign import StarSignDataset
from downstream.datasets.Briggs import Briggs
from downstream.datasets.MoralFoundation import MoralFoundation
from downstream.datasets.PoliticalLeaning import PoliticalLeaning
from downstream.datasets.BigFive import BigFive

def train(
    dataset_name: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda"
) -> None:
    """Executes the training loop and serializes the model.

    Args:
        dataset_name (str): Target dataset to load and train against.
        epochs (int): Number of passes over the dataset.
        batch_size (int): Samples per gradient update.
        lr (float): Learning rate.
        device (str): Compute device.
    """
    if dataset_name == "briggs":
        dataset = Briggs()
        out_classes = 16
    elif dataset_name == "starsign":
        dataset = StarSignDataset()
        out_classes = len(dataset.signs)
    elif dataset_name == "moralfoundation":
        dataset = MoralFoundation()
        out_classes = dataset.num_classes
    elif dataset_name == "politicalleaning":
        dataset = PoliticalLeaning()
        out_classes = dataset.num_classes
    elif dataset_name == "bigfive":
        dataset = BigFive()
        out_classes = dataset.num_classes
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    in_features = 384

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CoolProjectionHead(in_features=in_features, out_classes=out_classes).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device_obj), batch_y.to(device_obj)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # serialization
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{dataset_name}.pt"

    torch.save(model.state_dict(), out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train(dataset_name=args.dataset, epochs=args.epochs)