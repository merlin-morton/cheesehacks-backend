import os
import math
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_from_disk

from model import SharedEncoderBinaryHeads


class EthicsEmbDataset(Dataset):
    def __init__(self, hf_split, require_cols: Optional[List[str]] = None):
        self.ds = hf_split
        self.require_cols = require_cols or ["embedding", "label"]

        cols = set(self.ds.column_names)
        missing = [c for c in self.require_cols if c not in cols]
        if missing:
            raise ValueError(
                f"Dataset split is missing required columns {missing}. "
                f"Has columns: {self.ds.column_names}"
            )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.ds[idx]

        emb = row["embedding"]
        if not isinstance(emb, (list, tuple)):
            raise TypeError(f"Expected 'embedding' to be list/tuple, got {type(emb)} at idx={idx}")

        x = torch.tensor(emb, dtype=torch.float32)  # [D]

        lab = row["label"]
        # robust conversion: bool/int/float/str
        if isinstance(lab, bool):
            y_val = 1.0 if lab else 0.0
        elif isinstance(lab, (int, float)):
            y_val = float(lab)
        elif isinstance(lab, str):
            y_val = float(int(lab.strip()))
        else:
            raise TypeError(f"Unsupported label type {type(lab)} at idx={idx}: {lab}")

        # clamp just in case (prevents weird 2/ -1 values from crashing accuracy calc)
        y_val = 1.0 if y_val >= 0.5 else 0.0

        y = torch.tensor(y_val, dtype=torch.float32)  # []
        return x, y


def get_split_for_eval(ds_dict):
    
    if "test" in ds_dict:
        return "test"
    if "validation" in ds_dict:
        return "validation"
    return None


@torch.no_grad()
def eval_task(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

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

    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    stats: Dict[str, Dict[str, float]] = {}

    for task, loader in train_loaders.items():
        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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

    train_loaders: Dict[str, DataLoader] = {}
    eval_loaders: Dict[str, DataLoader] = {}

    input_dim = None

    for task in args.tasks:
        path = os.path.join(args.data_dir, f"ethics_{task}_with_embeddings")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset at {path}. Run save_embeddings.py first to create it."
            )

        ds = load_from_disk(path)

        if "train" not in ds:
            raise ValueError(f"{path} must contain a 'train' split. Found splits: {list(ds.keys())}")

        eval_split = get_split_for_eval(ds)
        if eval_split is None:
            raise ValueError(
                f"{path} has no 'test' or 'validation' split. Found splits: {list(ds.keys())}"
            )

        train_cols = ds["train"].column_names
        eval_cols = ds[eval_split].column_names
        for split_name, cols in [("train", train_cols), (eval_split, eval_cols)]:
            if "embedding" not in cols or "label" not in cols:
                raise ValueError(
                    f"{path}/{split_name} missing 'embedding' or 'label'. "
                    f"Columns: {cols}"
                )

        if input_dim is None:
            first = ds["train"][0]["embedding"]
            input_dim = len(first)

        train_ds = EthicsEmbDataset(ds["train"])
        eval_ds = EthicsEmbDataset(ds[eval_split])

        train_loaders[task] = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        eval_loaders[task] = DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        print(f"[{task}] splits: {list(ds.keys())} | train cols: {train_cols} | eval split: {eval_split}")

    assert input_dim is not None
    model = SharedEncoderBinaryHeads(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        tasks=args.tasks,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print(f"\nDevice: {device}")
    print(f"Tasks: {args.tasks}")
    print(f"Input dim: {input_dim}, latent dim: {args.latent_dim}")
    print("Training is grouped in task-batches each epoch (one task at a time).")

    best_avg_acc = -math.inf

    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch_grouped_by_task(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        eval_stats = {t: eval_task(model, eval_loaders[t], t, device) for t in args.tasks}
        avg_acc = sum(eval_stats[t]["acc"] for t in args.tasks) / len(args.tasks)

        print(f"\nEpoch {epoch}/{args.epochs}")
        for t in args.tasks:
            tr = train_stats[t]
            ev = eval_stats[t]
            print(
                f"  [{t:14s}] "
                f"train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
                f"eval  loss {ev['loss']:.4f} acc {ev['acc']:.3f}"
            )
        print(f"  avg eval acc: {avg_acc:.3f}")

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