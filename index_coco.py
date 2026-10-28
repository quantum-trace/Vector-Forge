from pathlib import Path
import json
import os
import sys

import open_clip
import torch
from PIL import Image

PROJECT = Path.home() / "vectorforge-image-search"
IMAGE_DIR = PROJECT / "datasets" / "coco" / "images"
DB_DIR = PROJECT / "data" / "images"
MAPPING_FILE = PROJECT / "image_ids.json"

VECTORFORGE_ROOT = Path(
    "/mnt/c/Users/pprii/Downloads/VectorForge/VectorForge"
)

sys.path.insert(
    0, str(VECTORFORGE_ROOT / "bindings" / "python")
)
os.environ["VECTORFORGE_LIB"] = str(
    VECTORFORGE_ROOT / "build" / "libvectorforge.so"
)

import vectorforge as vf

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    with MAPPING_FILE.open() as f:
        image_ids = json.load(f)

    paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED
    )

    print(f"Found {len(paths)} COCO images.")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    model.eval()

    next_id = max(image_ids.values(), default=0) + 1
    added = 0
    failed = 0

    with vf.Collection(
        dim=512,
        metric="cosine",
        path=str(DB_DIR),
    ) as db:
        with torch.no_grad():
            for index, path in enumerate(paths, start=1):
                key = f"coco/{path.name}"

                if key in image_ids:
                    continue

                try:
                    with Image.open(path) as source:
                        tensor = preprocess(
                            source.convert("RGB")
                        ).unsqueeze(0)

                    vector = model.encode_image(tensor)
                    vector = vector / vector.norm(
                        dim=-1, keepdim=True
                    )

                    image_id = next_id
                    db.upsert(
                        image_id,
                        vector.squeeze(0).numpy(),
                        {
                            "filename": path.name,
                            "source": "coco",
                            "type": "image",
                        },
                    )

                    image_ids[key] = image_id
                    next_id += 1
                    added += 1

                except Exception as exc:
                    failed += 1
                    print(f"Skipped {path.name}: {exc}")

                if index % 25 == 0:
                    print(
                        f"Progress: {index}/{len(paths)} | "
                        f"Indexed: {added} | Failed: {failed}"
                    )

        db.checkpoint()

    MAPPING_FILE.write_text(
        json.dumps(image_ids, indent=2) + "\n"
    )

    print("\nCOCO indexing complete!")
    print(f"New images indexed: {added}")
    print(f"Failed images: {failed}")
    print(f"Total mapped images: {len(image_ids)}")


if __name__ == "__main__":
    main()
