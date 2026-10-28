from pathlib import Path
import os
import sys

import open_clip
import torch
from PIL import Image

PROJECT = Path.home() / "vectorforge-image-search"
IMAGE_DIR = PROJECT / "images"
DB_DIR = PROJECT / "data" / "images"

VECTORFORGE_ROOT = Path("/mnt/c/Users/pprii/Downloads/VectorForge/VectorForge")
sys.path.insert(0, str(VECTORFORGE_ROOT / "bindings" / "python"))
os.environ["VECTORFORGE_LIB"] = str(VECTORFORGE_ROOT / "build" / "libvectorforge.so")

import vectorforge as vf

def main():
    filename = input("Enter image filename (e.g. bus.jpg): ").strip()
    image_path = (IMAGE_DIR / filename).resolve()

    if image_path.parent != IMAGE_DIR.resolve() or not image_path.is_file():
        print(f"Image not found in {IMAGE_DIR}")
        return

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vector = model.encode_image(tensor)
        vector = vector / vector.norm(dim=-1, keepdim=True)

    with vf.Collection(dim=512, metric="cosine", path=str(DB_DIR)) as db:
        results = db.search(vector.numpy(), k=min(5, len(db)))

    print(f"\nSimilar images for: {filename}")
    if not results:
        print("No matching images found.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result}")

if __name__ == "__main__":
    main()
