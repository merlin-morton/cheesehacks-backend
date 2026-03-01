import argparse
from pathlib import Path
from downstream.model import CoolProjectionHead

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from tqdm import tqdm

from downstream.datasets.StarSign import StarSignDataset
from downstream.datasets.Briggs import Briggs
from downstream.datasets.MoralFoundation import MoralFoundation
from downstream.datasets.PoliticalLeaning import PoliticalLeaning
from downstream.datasets.BigFive import BigFive

from sentence_transformers import SentenceTransformer

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
    encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=str(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    if dataset_name == "briggs":
        dataset = Briggs(encoder)
        out_classes = 16

        test_dataset = Briggs(encoder, split="test")
    elif dataset_name == "starsign":
        full_dataset = StarSignDataset(encoder=encoder)

        # split dataset
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        out_classes = len(dataset.signs)
    elif dataset_name == "moralfoundation":
        dataset = MoralFoundation(encoder)
        out_classes = dataset.num_classes
        test_dataset = MoralFoundation(encoder, split="test")
    elif dataset_name == "politicalleaning":
        dataset = PoliticalLeaning(encoder)
        out_classes = dataset.num_classes
        test_dataset = PoliticalLeaning(encoder)
    elif dataset_name == "bigfive":
        dataset = BigFive(encoder)
        out_classes = dataset.num_classes
        test_dataset = BigFive(encoder)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    in_features = 384

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CoolProjectionHead(in_features=in_features, out_classes=out_classes).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # cross entropy bc we are boringfr54
    criterion = nn.CrossEntropyLoss()

    # training loop
    model.train()
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device_obj), batch_y.to(device_obj)

            optimizer.zero_grad()
            logits = model(batch_x)

            # loss calculation
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            # online metrics
            epoch_loss += loss.item()

            # accuracy: [B, C] -> [B]
            _, predicted = torch.max(logits.data, 1)
            epoch_total += batch_y.size(0)
            epoch_correct += (predicted == batch_y).sum().item()

        # epoch logging
        avg_loss = epoch_loss / len(loader)
        acc = 100 * epoch_correct / epoch_total
        print(
            f"epoch [{epoch + 1}/{epochs}] - loss: {avg_loss:.4f} - acc: {acc:.2f}%")

    # testing
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device_obj), batch_y.to(device_obj)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            test_loss += loss.item()

            # acc
            _, predicted = torch.max(logits.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()

    # final metrics
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    print(f"test - loss: {avg_test_loss:.4f} - acc: {test_acc:.2f}%")

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