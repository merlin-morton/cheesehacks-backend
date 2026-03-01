import argparse
import json
from typing import List, Dict, Any, Optional

import torch
from sentence_transformers import SentenceTransformer

from model import SharedEncoderBinaryHeads


DEFAULT_TASKS = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]


def load_checkpoint_model(ckpt_path: str, device: torch.device) -> SharedEncoderBinaryHeads:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    input_dim = int(ckpt["input_dim"])
    latent_dim = int(ckpt["latent_dim"])
    tasks = list(ckpt.get("tasks", DEFAULT_TASKS))
    dropout = float(ckpt.get("dropout", 0.1))

    model = SharedEncoderBinaryHeads(
        input_dim=input_dim,
        latent_dim=latent_dim,
        tasks=tasks,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def classify_batch(
    sentences: List[str],
    model: SharedEncoderBinaryHeads,
    embedder: SentenceTransformer,
    tasks: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    return_probs: bool = True,
) -> Dict[str, Any]:
    if tasks is None:
        tasks = list(model.heads.keys())
    if device is None:
        device = next(model.parameters()).device

    emb = embedder.encode(
        sentences,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        convert_to_tensor=True,   # gives torch.Tensor directly
        show_progress_bar=False,
    )  # [B, D] on embedder device
    emb = emb.to(device)

    z = model.encoder(emb)  # [B, H]

    logits_per_task = []
    for t in tasks:
        if t not in model.heads:
            raise KeyError(f"Task '{t}' not in model.heads. Available: {list(model.heads.keys())}")
        logits_t = model.heads[t](z).squeeze(-1)  # [B]
        logits_per_task.append(logits_t)

    logits = torch.stack(logits_per_task, dim=1)  # [B, K]

    out: Dict[str, Any] = {
        "tasks": tasks,
        "sentences": sentences,
        "logits": logits.detach().cpu().tolist(),
    }
    if return_probs:
        probs = torch.sigmoid(logits)
        out["probs"] = probs.detach().cpu().tolist()

    return out


def read_sentences_from_args(sentences: List[str], file_path: Optional[str]) -> List[str]:
    sents = list(sentences)

    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sents.append(line)

    if len(sents) == 1:
        one = sents[0].strip()
        if (one.startswith("[") and one.endswith("]")) or (one.startswith('"') and one.endswith('"')):
            try:
                maybe = json.loads(one)
                if isinstance(maybe, list) and all(isinstance(x, str) for x in maybe):
                    sents = maybe
            except Exception:
                pass

    if not sents:
        raise ValueError("No sentences provided. Use positional SENTENCES and/or --file.")

    return sents


def main():
    parser = argparse.ArgumentParser(description="Classify sentences across all moral heads.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/shared_encoder_heads.pt",
                        help="Path to checkpoint saved by train.py")
    parser.add_argument("--embedder", type=str, default="BAAI/bge-base-en-v1.5",
                        help="SentenceTransformer model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_normalize", action="store_true",
                        help="Disable embedding normalization")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Optional subset of tasks/heads to run. Default: all heads in checkpoint.")

    parser.add_argument("--file", type=str, default=None,
                        help="Optional text file with one sentence per line")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON (default). If false, prints a readable table.")
    parser.add_argument("sentences", nargs="*", help="Sentences to classify (or pass --file).")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load model + embedder
    model = load_checkpoint_model(args.ckpt, device=device)
    embedder = SentenceTransformer(args.embedder, device=str(device))

    sentences = read_sentences_from_args(args.sentences, args.file)

    result = classify_batch(
        sentences=sentences,
        model=model,
        embedder=embedder,
        tasks=args.tasks,
        device=device,
        batch_size=args.batch_size,
        normalize_embeddings=(not args.no_normalize),
        return_probs=True,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        tasks = result["tasks"]
        probs = result.get("probs", None)

        print("Tasks:", ", ".join(tasks))
        print()
        for i, s in enumerate(result["sentences"]):
            row = probs[i] if probs is not None else result["logits"][i]
            vals = "  ".join([f"{t[:4]}={row[j]:.3f}" for j, t in enumerate(tasks)])
            print(f"- {s}\n  {vals}\n")


if __name__ == "__main__":
    main()