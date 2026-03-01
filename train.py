# train_model.py
import os
import math
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_from_disk

from model import SharedEncoderBinaryHeads   



# ---- Dataset wrapper ----
class EthicsEmbDataset(Dataset):
    """
    Expects HuggingFace dataset with columns:
      - "embedding": list[float] length D
      - "label": binary (0/1 or False/True)
    """
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.ds[idx]
        x = torch.tensor(row["embedding"], dtype=torch.float32)  # [D]
        # label may be bool/int
        y = torch.tensor(float(row["label"]), dtype=torch.float32)  # []
        return x, y

def get_model(input_dim: int, latent_dim: int, tasks: List[str], dropout: float) -> nn.Module:

    return SharedEncoderBinaryHeads(input_dim=input_dim, latent_dim=latent_dim, tasks=tasks, dropout=dropout)


# ---- Train / Eval helpers ----
@torch.no_grad()
def eval_task(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x, task).squeeze(-1)  # [B]
        loss = bce(logits, y)
        total_loss += float(loss.item())

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += int((preds == y).sum().item())
        total += y.numel()

    return {
        "loss": total_loss / max(1, total),
        "acc": correct / max(1, total),
        "n": float(total),
    }


def train_epoch_grouped_by_task(
    model: nn.Module,
    train_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    One epoch where we train in *task-batches*:
      for task in tasks:
        for batch in task_loader:
          update(model on that task)
    """
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    stats: Dict[str, Dict[str, float]] = {}

    for task, loader in train_loaders.items():
        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, task).squeeze(-1)  # [B]
            loss = loss_fn(logits, y)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            running_loss += float(loss.item()) * y.numel()
            total += y.numel()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += int((preds == y).sum().item())

        stats[task] = {
            "loss": running_loss / max(1, total),
            "acc": correct / max(1, total),
            "n": float(total),
        }

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing saved HF datasets")
    parser.add_argument("--tasks", nargs="+", default=["commonsense", "deontology", "justice", "utilitarianism", "virtue"])
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--save_path", type=str, default="checkpoints/shared_encoder_heads.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load datasets saved by save_embeddings.py (e.g., data/ethics_deontology_with_embeddings) :contentReference[oaicite:2]{index=2}
    train_loaders: Dict[str, DataLoader] = {}
    test_loaders: Dict[str, DataLoader] = {}

    input_dim = None

    for task in args.tasks:
        path = os.path.join(args.data_dir, f"ethics_{task}_with_embeddings")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset at {path}. "
                f"Run save_embeddings.py first to create it."
            )

        ds = load_from_disk(path)
        if "train" not in ds or "test" not in ds:
            raise ValueError(f"{path} must contain train and test splits.")

        # Determine embedding dimension from first element
        if input_dim is None:
            first = ds["train"][0]["embedding"]
            input_dim = len(first)

        train_ds = EthicsEmbDataset(ds["train"])
        test_ds = EthicsEmbDataset(ds["test"])

        train_loaders[task] = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        test_loaders[task] = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    assert input_dim is not None
    model = SharedEncoderBinaryHeads(input_dim=input_dim, latent_dim=args.latent_dim, tasks=args.tasks, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print(f"Device: {device}")
    print(f"Tasks: {args.tasks}")
    print(f"Input dim: {input_dim}, latent dim: {args.latent_dim}")
    print(f"Training is grouped in task-batches each epoch (one task at a time).")

    best_avg_acc = -math.inf

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch_grouped_by_task(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        # Evaluate each task
        eval_stats = {t: eval_task(model, test_loaders[t], t, device) for t in args.tasks}
        avg_acc = sum(eval_stats[t]["acc"] for t in args.tasks) / len(args.tasks)

        print(f"\nEpoch {epoch}/{args.epochs}")
        for t in args.tasks:
            tr = train_stats[t]
            ev = eval_stats[t]
            print(
                f"  [{t:14s}] "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
                f"test loss {ev['loss']:.4f} acc {ev['acc']:.3f}"
            )
        print(f"  avg test acc: {avg_acc:.3f}")

        # Save best checkpoint
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "latent_dim": args.latent_dim,
                    "tasks": args.tasks,
                    "dropout": args.dropout,
                },
                args.save_path,
            )
            print(f"  ✅ saved best checkpoint to {args.save_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()