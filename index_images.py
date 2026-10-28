from pathlib import Path
import json
import os
import sys

import open_clip
import torch
from PIL import Image

PROJECT = Path.home() / "vectorforge-image-search"
IMAGE_DIR = PROJECT / "images"
DB_DIR = PROJECT / "data" / "images"
MAPPING_FILE = PROJECT / "image_ids.json"

VECTORFORGE_ROOT = Path("/mnt/c/Users/pprii/Downloads/VectorForge/VectorForge")
sys.path.insert(0, str(VECTORFORGE_ROOT / "bindings" / "python"))
os.environ["VECTORFORGE_LIB"] = str(VECTORFORGE_ROOT / "build" / "libvectorforge.so")

import vectorforge as vf

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    if MAPPING_FILE.exists():
        image_ids = json.loads(MAPPING_FILE.read_text())
    else:
        image_ids = {}

    paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    if not paths:
        print(f"No supported images found in {IMAGE_DIR}")
        return

    next_id = max(image_ids.values(), default=0) + 1
    for path in paths:
        if path.name not in image_ids:
            image_ids[path.name] = next_id
            next_id += 1

    MAPPING_FILE.write_text(json.dumps(image_ids, indent=2) + "\n")

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device="cpu"
    )
    model.eval()

    with vf.Collection(dim=512, metric="cosine", path=str(DB_DIR)) as db:
        with torch.no_grad():
            for path in paths:
                try:
                    with Image.open(path) as source:
                        tensor = preprocess(source.convert("RGB")).unsqueeze(0)

                    vector = model.encode_image(tensor)
                    vector = vector / vector.norm(dim=-1, keepdim=True)
                    image_id = image_ids[path.name]

                    db.upsert(
                        image_id,
                        vector.squeeze(0).numpy(),
                        {"filename": path.name, "type": "image"},
                    )
                    print(f"Indexed: {path.name} (ID {image_id})")

                except Exception as exc:
                    print(f"Skipped {path.name}: {exc}")

        db.checkpoint()
        print(f"\nDatabase contains {len(db)} vectors.")
        print(f"ID mapping saved to {MAPPING_FILE}")

if __name__ == "__main__":
    main()
