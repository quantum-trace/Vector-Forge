from pathlib import Path
import json
import os
import sys

import open_clip
import torch

PROJECT = Path.home() / "vectorforge-image-search"
DB_DIR = PROJECT / "data" / "images"
MAPPING_FILE = PROJECT / "image_ids.json"

VECTORFORGE_ROOT = Path(
    "/mnt/c/Users/pprii/Downloads/VectorForge/VectorForge"
)
sys.path.insert(0, str(VECTORFORGE_ROOT / "bindings" / "python"))
os.environ["VECTORFORGE_LIB"] = str(
    VECTORFORGE_ROOT / "build" / "libvectorforge.so"
)

import vectorforge as vf


def main():
    query = input("Describe the image you want to find: ").strip()

    if not query:
        print("Please enter a search query.")
        return

    image_ids = json.loads(MAPPING_FILE.read_text())
    filenames = {image_id: name for name, image_id in image_ids.items()}

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    with torch.no_grad():
        tokens = tokenizer([query])
        vector = model.encode_text(tokens)
        vector = vector / vector.norm(dim=-1, keepdim=True)

    with vf.Collection(dim=512, metric="cosine", path=str(DB_DIR)) as db:
        results = db.search(vector.numpy(), k=min(5, len(db)))

    print(f"\nSearch results for: {query}")

    if not results:
        print("No matching images found.")
        return

    for rank, result in enumerate(results, start=1):
        image_id, distance = result
        filename = filenames.get(image_id, "unknown image")
        print(f"{rank}. {filename} | Distance: {distance:.4f}")


if __name__ == "__main__":
    main()
