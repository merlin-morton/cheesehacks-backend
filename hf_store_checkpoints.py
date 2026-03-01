#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

def eprint(*args):
    print(*args, file=sys.stderr, flush=True)

def main():
    eprint("==> Starting upload_checkpoints.py")

    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", required=True, help="e.g. your-username/your-repo")
    p.add_argument("--checkpoint_dir", default="checkpoints", help="Local folder to upload")
    p.add_argument("--message", default="Upload checkpoints", help="Commit message")
    p.add_argument("--private", action="store_true", help="Create repo as private if it doesn't exist")
    p.add_argument("--branch", default=None, help="Branch name (default: main)")
    p.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    p.add_argument("--include", action="append", default=[], help="Glob to include (can repeat)")
    p.add_argument("--exclude", action="append", default=[], help="Glob to exclude (can repeat)")
    p.add_argument("--delete", action="store_true",
                   help="Delete files on Hub not present locally (only if your hub version supports it)")
    args = p.parse_args()

    eprint("==> Parsed args:", args)

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        eprint("ERROR: No token found. Set HF_TOKEN or pass --token.")
        sys.exit(2)
    eprint("==> Token found (hidden). Length:", len(token))

    ckpt_dir = Path(args.checkpoint_dir)
    eprint("==> checkpoint_dir:", ckpt_dir.resolve())
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        eprint(f"ERROR: checkpoint_dir not found or not a directory: {ckpt_dir}")
        sys.exit(3)

    # Show a few files so we know we're uploading something
    files = [p for p in ckpt_dir.rglob("*") if p.is_file()]
    eprint(f"==> Found {len(files)} files under {ckpt_dir}")
    for f in files[:10]:
        eprint("   -", f.relative_to(ckpt_dir))

    eprint("==> Importing huggingface_hub...")
    from huggingface_hub import create_repo, upload_folder
    eprint("==> Imported huggingface_hub OK")

    eprint("==> Ensuring repo exists:", args.repo_id)
    create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
        token=token,
    )
    eprint("==> Repo ready")

    allow_patterns = args.include or ["*.pt", "*.pth", "*.bin", "*.safetensors", "*.json", "*.txt", "*.md"]
    ignore_patterns = args.exclude or ["*.tmp", "*.log", "*.bak", "__pycache__/*", "*.ipynb_checkpoints/*"]

    # ---- Upload (compatible with old/new huggingface_hub) ----
    eprint("==> Uploading folder now...")

    upload_kwargs = dict(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(ckpt_dir),
        path_in_repo="checkpoints",
        commit_message=args.message,
        token=token,
        revision=args.branch,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    try:
        # Newer versions support delete=...
        upload_folder(**upload_kwargs, delete=args.delete)
    except TypeError as e:
        # Older versions don't accept delete
        if "unexpected keyword argument 'delete'" in str(e):
            if args.delete:
                eprint("WARN: huggingface_hub doesn't support --delete in this environment; uploading without deleting remote files.")
            upload_folder(**upload_kwargs)
        else:
            raise

    url = f"https://huggingface.co/{args.repo_id}/tree/{args.branch or 'main'}/checkpoints"
    print(f"✅ Uploaded successfully: {url}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        eprint("FATAL ERROR:", repr(ex))
        raise