from pathlib import Path
from urllib.parse import urlparse
import json
import time

import pyarrow.parquet as pq
import requests
from PIL import Image
from io import BytesIO

PROJECT = Path.home() / "vectorforge-image-search"
PARQUET_FILE = PROJECT / "datasets" / "coco" / "coco_sample.parquet"
OUTPUT_DIR = PROJECT / "datasets" / "coco" / "images"
MANIFEST_FILE = PROJECT / "datasets" / "coco" / "download_manifest.json"

LIMIT = 1000
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    table = pq.read_table(
        PARQUET_FILE,
        columns=["file_name", "flickr_url"]
    ).slice(0, LIMIT)

    session = requests.Session()
    session.headers.update({"User-Agent": "VectorForge-ImageSearch/1.0"})

    manifest = {}
    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text())

    success = 0
    failed = 0

    for index, row in enumerate(table.to_pylist(), start=1):
        filename = Path(row["file_name"]).name
        url = row["flickr_url"]

        if not url:
            failed += 1
            continue

        url = url.replace("http://", "https://", 1)
        destination = OUTPUT_DIR / filename

        if destination.exists():
            try:
                with Image.open(destination) as image:
                    image.verify()
                success += 1
                continue
            except Exception:
                destination.unlink(missing_ok=True)

        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()

            with Image.open(BytesIO(response.content)) as image:
                image_format = image.format
                image.verify()

            if image_format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Unsupported image format")

            destination.write_bytes(response.content)
            manifest[filename] = {
                "source_url": url,
                "original_file_name": row["file_name"],
            }
            success += 1

        except Exception as exc:
            failed += 1
            print(f"Failed [{index}] {filename}: {exc}")

        if index % 50 == 0:
            MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
            print(f"Progress: {index}/{len(table)} | downloaded/valid: {success} | failed: {failed}")

        time.sleep(0.1)

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))
    print(f"\nFinished. Valid images: {success} | Failed: {failed}")
    print(f"Images folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
