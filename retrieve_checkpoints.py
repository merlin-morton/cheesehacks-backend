from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="Praneet-P/ethics-multihead-model",
    repo_type="model",
)

print("Downloaded to:", local_dir)