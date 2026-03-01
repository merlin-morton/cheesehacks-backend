from datasets import load_dataset
from sentence_transformers import SentenceTransformer

ethics = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]

for ethic in ethics:


    ds = load_dataset("hendrycks/ethics", ethic)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    TEXT_COL = "input"   # change if your dataset uses a different field name

    def embed_batch(batch):
        vecs = model.encode(
            batch[TEXT_COL],
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return {"embedding": vecs.tolist()}

    # Apply to both train and test
    for split in ds.keys():  # typically "train", "test"
        ds[split] = ds[split].map(embed_batch, batched=True, batch_size=256)

    # Save everything locally (train+test+embeddings)
    ds.save_to_disk(f"data/ethics_{ethic}_with_embeddings")