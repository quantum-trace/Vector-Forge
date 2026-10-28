from pathlib import Path
import json
import os
import sys

import open_clip
import streamlit as st
import torch
from PIL import Image

PROJECT = Path.home() / "vectorforge-image-search"
MAPPING_FILE = PROJECT / "image_ids.json"
DB_DIR = PROJECT / "data" / "images"

VECTORFORGE_ROOT = Path(
    "/mnt/c/Users/pprii/Downloads/VectorForge/VectorForge"
)
sys.path.insert(0, str(VECTORFORGE_ROOT / "bindings" / "python"))
os.environ["VECTORFORGE_LIB"] = str(
    VECTORFORGE_ROOT / "build" / "libvectorforge.so"
)

import vectorforge as vf

st.set_page_config(
    page_title="VectorForge Image Search",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 VectorForge")
st.caption("AI-powered image search using CLIP + VectorForge")

@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer

@st.cache_data
def load_mapping():
    return json.loads(MAPPING_FILE.read_text())

def embed_text(text, model, tokenizer):
    tokens = tokenizer([text])
    with torch.no_grad():
        vector = model.encode_text(tokens)
        vector = vector / vector.norm(dim=-1, keepdim=True)
    return vector.squeeze(0).numpy()

def embed_image(image, model, preprocess):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        vector = model.encode_image(tensor)
        vector = vector / vector.norm(dim=-1, keepdim=True)
    return vector.squeeze(0).numpy()

def get_image_path(key):
    if key.startswith("coco/"):
        return PROJECT / "datasets" / "coco" / "images" / key.split("/", 1)[1]
    return PROJECT / "images" / key

def show_results(results, mapping):
    id_to_key = {int(image_id): key for key, image_id in mapping.items()}
    columns = st.columns(4)

    for index, result in enumerate(results):
        image_id, distance = int(result[0]), float(result[1])
        key = id_to_key.get(image_id)

        if key is None:
            continue

        image_path = get_image_path(key)
        with columns[index % 4]:
            if image_path.exists():
                st.image(str(image_path), width="stretch")
                st.markdown(f"**{key.split('/')[-1]}**")
                st.caption(f"Rank #{index + 1} · Distance: {distance:.4f}")
            else:
                st.warning(f"Image file missing: {key}")

mapping = load_mapping()
model, preprocess, tokenizer = load_model()

with st.sidebar:
    st.header("Search settings")
    result_count = st.slider("Number of results", 1, 20, 8)
    st.metric("Indexed images", len(mapping))

mode = st.radio(
    "Choose search mode",
    ["Text Search", "Image Search"],
    horizontal=True,
)

if mode == "Text Search":
    query = st.text_input(
        "Describe the image you want to find",
        placeholder="e.g. a dog playing in the grass",
    )

    if st.button("Search text", type="primary") and query.strip():
        vector = embed_text(query.strip(), model, tokenizer)

        with vf.Collection(
            dim=512,
            metric="cosine",
            path=str(DB_DIR),
        ) as db:
            results = db.search(vector, k=result_count)

        st.subheader(f"Results for: {query}")
        st.caption("Lower cosine distance means a closer match.")
        show_results(results, mapping)

else:
    uploaded_file = st.file_uploader(
        "Upload an image to find visually similar images",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.image(uploaded_image, caption="Your query image", width=300)

            if st.button("Find similar images", type="primary"):
                vector = embed_image(uploaded_image, model, preprocess)

                with vf.Collection(
                    dim=512,
                    metric="cosine",
                    path=str(DB_DIR),
                ) as db:
                    results = db.search(vector, k=result_count)

                st.subheader("Visually similar images")
                st.caption("Lower cosine distance means a closer match.")
                show_results(results, mapping)

        except Exception as exc:
            st.error(f"Could not process image: {exc}")

st.divider()
st.caption("VectorForge · CLIP embeddings · Vector similarity search")
