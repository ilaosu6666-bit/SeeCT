"""Upload PNG images to HF Space via API (bypasses Xet restriction)."""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
repo_id = "ILAOSU/seeCT"
repo_type = "space"

base = Path(".")
png_files = sorted(list(base.glob("cases/**/*.png")) + list(base.glob("loss_fig/**/*.png")))
print(f"Uploading {len(png_files)} PNG files...")

for i, p in enumerate(png_files):
    path_in_repo = str(p).replace("\\", "/")
    if i < 5 or i % 20 == 0:
        print(f"  [{i+1}/{len(png_files)}] {path_in_repo}")
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )

print(f"Done! Uploaded {len(png_files)} files.")
