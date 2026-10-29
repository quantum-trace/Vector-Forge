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

TARGETS = [
    "000000133821.jpg",
    "000000212342.jpg",
    "000000467593.jpg",
    "000000555915.jpg",
]


def main():
    image_ids = json.loads(MAPPING_FILE.read_text())

    paths = [IMAGE_DIR / name for name in TARGETS]
    missing = [path.name for path in paths if not path.is_file()]

    if missing:
        print("Missing images:", ", ".join(missing))
        return

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    model.eval()

    next_id = max(image_ids.values(), default=0) + 1
    added = 0

    with vf.Collection(
        dim=512,
        metric="cosine",
        path=str(DB_DIR),
    ) as db:
        with torch.no_grad():
            for path in paths:
                key = f"coco/{path.name}"

                if key in image_ids:
                    print(f"Already mapped, skipping: {key}")
                    continue

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
                print(f"Indexed: {key} (ID {image_id})")

        db.checkpoint()

    MAPPING_FILE.write_text(
        json.dumps(image_ids, indent=2) + "\n"
    )

    print(f"\nNew images indexed: {added}")
    print(f"Total mapped images: {len(image_ids)}")


if __name__ == "__main__":
    main()
