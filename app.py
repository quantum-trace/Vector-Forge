
from pathlib import Path
from html import unescape
import re
import json
import mimetypes
import os
import sys

import open_clip
import requests
import streamlit as st
import torch
from PIL import Image

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT = Path(__file__).resolve().parent
MAPPING_FILE = PROJECT / "image_ids.json"
THEME_SETTINGS_FILE = PROJECT / "theme_settings.json"
DB_DIR = PROJECT / "data" / "images"

# Override these paths through environment variables when needed.
VECTORFORGE_ROOT = Path(
    os.environ.get(
        "VECTORFORGE_ROOT",
        str(PROJECT / "VectorForge"),
    )
).expanduser()

VECTORFORGE_BINDINGS = Path(
    os.environ.get(
        "VECTORFORGE_BINDINGS",
        str(VECTORFORGE_ROOT / "bindings" / "python"),
    )
).expanduser()

VECTORFORGE_LIBRARY = Path(
    os.environ.get(
        "VECTORFORGE_LIB",
        str(VECTORFORGE_ROOT / "build" / "libvectorforge.so"),
    )
).expanduser()

sys.path.insert(0, str(VECTORFORGE_BINDINGS))
os.environ["VECTORFORGE_LIB"] = str(VECTORFORGE_LIBRARY)

from build_vectorforge import ensure_vectorforge_built

VECTORFORGE_LIBRARY = ensure_vectorforge_built()
os.environ["VECTORFORGE_LIB"] = str(VECTORFORGE_LIBRARY)

import vectorforge as vf



@st.cache_data(ttl=300, show_spinner=False)
def search_wikimedia_images(query, limit=8):
    """Search Wikimedia Commons and return image metadata."""
    endpoint = "https://commons.wikimedia.org/w/api.php"

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": min(max(int(limit), 1), 20),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 480,
        "format": "json",
        "origin": "*",
    }

    headers = {
        "User-Agent": "VectorForgeImageSearch/1.0 (image search app)"
    }

    try:
        response = requests.get(
            endpoint,
            params=params,
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()

        results = []
        seen_titles = set()
        seen_urls = set()

        query_terms = {
            term.casefold()
            for term in re.findall(r"[a-zA-Z0-9]+", query)
            if len(term) > 1
        }

        # Expand common fruit names with botanical and descriptive synonyms.
        query_expansions = {
            "strawberry": {"strawberry", "strawberries", "fragaria", "berry", "fruit"},
            "strawberries": {"strawberry", "strawberries", "fragaria", "berry", "fruit"},
            "apple": {"apple", "malus", "fruit"},
            "banana": {"banana", "musa", "fruit"},
            "orange": {"orange", "citrus", "fruit"},
            "mango": {"mango", "mangifera", "fruit"},
            "grape": {"grape", "vitis", "fruit"},
            "peach": {"peach", "prunus", "fruit"},
            "pear": {"pear", "pyrus", "fruit"},
            "cherry": {"cherry", "prunus", "fruit"},
            "blueberry": {"blueberry", "vaccinium", "berry", "fruit"},
            "raspberry": {"raspberry", "rubus", "berry", "fruit"},
            "lemon": {"lemon", "citrus", "fruit"},
        }

        expanded_terms = set(query_terms)
        for term in query_terms:
            expanded_terms.update(query_expansions.get(term, set()))

        is_fruit_query = any(
            term in query_expansions
            for term in query_terms
        )

        for item in payload.get("query", {}).get("pages", {}).values():
            info_list = item.get("imageinfo", [])
            if not info_list:
                continue

            info = info_list[0]
            metadata = info.get("extmetadata", {})
            title = item.get("title", "Untitled image")
            image_url = info.get("url", "")
            thumbnail = info.get("thumburl") or image_url

            normalized_title = re.sub(r"\\s+", " ", title).strip().casefold()
            normalized_url = image_url.split("?", 1)[0].rstrip("/").casefold()

            if (
                not image_url
                or normalized_title in seen_titles
                or normalized_url in seen_urls
            ):
                continue

            seen_titles.add(normalized_title)
            seen_urls.add(normalized_url)

            def metadata_value(key):
                value = metadata.get(key, {})
                return value.get("value", "") if isinstance(value, dict) else ""

            searchable_text = (
                f"{title} {metadata_value('ObjectName')} "
                f"{metadata_value('ImageDescription')}"
            ).casefold()

            title_lower = title.casefold()
            matched_terms = {
                term for term in expanded_terms if term in searchable_text
            }
            relevance_score = len(matched_terms) * 2

            # Strongly prefer explicit fruit/berry titles over leaf specimens.
            fruit_match = any(
                term in searchable_text
                for term in ("fruit", "berry", "strawberry", "strawberries")
            )
            leaf_match = any(
                term in searchable_text
                for term in (
                    "leaf", "leaves", "foliage", "herbarium",
                    "leaflet", "plant specimen"
                )
            )

            if is_fruit_query and fruit_match:
                relevance_score += 10
            if is_fruit_query and leaf_match:
                relevance_score -= 6
            if any(term in title_lower for term in query_terms):
                relevance_score += 3

            results.append({
                "title": title,
                "thumbnail": thumbnail,
                "image_url": image_url,
                "source_url": info.get("descriptionurl", ""),
                "license": metadata_value("LicenseShortName") or "Check source page",
                "artist": metadata_value("Artist") or "Not specified",
                "attribution": metadata_value("Attribution"),
                "_relevance_score": relevance_score,
            })

        results.sort(
            key=lambda result: result["_relevance_score"],
            reverse=True,
        )

        # Limit repetitive herbarium/leaf specimen results.
        # Keep fruit images unaffected by this grouping rule.
        filtered_results = []
        leaf_group_counts = {}
        seen_normalized_titles = set()

        for result in results:
            title = result.get("title", "")
            normalized_title = re.sub(
                r"\\bimg[-_ ]?\\d+\\b", "img-N", title.casefold()
            )
            normalized_title = re.sub(r"\\s+", " ", normalized_title).strip()

            if normalized_title in seen_normalized_titles:
                continue
            seen_normalized_titles.add(normalized_title)

            title_lower = title.casefold()
            is_leaf_specimen = any(
                term in title_lower
                for term in ("herbarium", "leaf specimen", "fragaria vesca")
            )
            is_fruit_result = any(
                term in title_lower
                for term in ("fruit", "strawberry", "strawberries", "berry")
            )

            if is_leaf_specimen and not is_fruit_result:
                group_key = re.sub(
                    r"\\bimg[-_ ]?\\d+\\b", "img-N", title_lower
                )
                count = leaf_group_counts.get(group_key, 0)
                if count >= 2:
                    continue
                leaf_group_counts[group_key] = count + 1

            filtered_results.append(result)

        for result in filtered_results:
            result.pop("_relevance_score", None)

        return filtered_results

    except (requests.RequestException, ValueError) as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_external_image(url, fallback_url="", referer=""):
    """Fetch original image bytes, retrying temporary failures and then fallback."""
    import time
    from urllib.parse import urlsplit, urlunsplit

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Referer": referer or "https://commons.wikimedia.org/",
    }

    def clean_image_url(value):
        if not value:
            return ""
        parts = urlsplit(value)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    last_error = None
    candidates = [
        clean_image_url(url),
        clean_image_url(fallback_url),
    ]

    for candidate in dict.fromkeys(candidates):
        if not candidate:
            continue

        for attempt in range(3):
            try:
                response = requests.get(
                    candidate,
                    headers=headers,
                    timeout=25,
                )

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        retry_after = response.headers.get("Retry-After", "")
                        try:
                            delay = min(float(retry_after), 10)
                        except (ValueError, TypeError):
                            delay = 2 ** attempt
                        time.sleep(max(delay, 1))
                        continue

                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if not content_type.startswith("image/"):
                    raise ValueError("The server did not return image data.")
                return response.content

            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status == 429 or (status and status >= 500):
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                        continue
                break

    raise ValueError(f"Could not fetch image from either URL: {last_error}")


def clean_metadata_text(value):
    """Convert HTML metadata into readable plain text."""
    if not value:
        return ""
    value = unescape(str(value))
    value = re.sub(r"<[^>]*>", " ", value)
    return " ".join(value.split()).strip()


def show_external_results(results):
    """Display external images with direct downloads and licence details."""
    if isinstance(results, dict) and "error" in results:
        st.warning(f"External image search failed: {results['error']}")
        return

    if not results:
        st.info("No external images found for this query.")
        return

    st.markdown("### Additional image results")
    st.caption(
        "External results · Licence and creator information is provided "
        "where available. Check the source page for full licence terms."
    )

    columns_per_row = 4

    for start in range(0, len(results), columns_per_row):
        row_results = results[start:start + columns_per_row]
        columns = st.columns(columns_per_row)

        for offset, item in enumerate(row_results):
            with columns[offset]:
                with st.container(border=True):
                    st.image(
                        item["thumbnail"],
                        use_container_width=True,
                    )
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"Licence: {item['license']}")

                    artist = clean_metadata_text(item.get("artist", ""))
                    if artist and artist != "Not specified":
                        st.caption(f"Artist: {artist}")

                    image_url = item.get("image_url") or item.get("thumbnail")
                    filename = clean_metadata_text(
                        item.get("title", "external_image")
                    ).replace("File:", "")
                    filename = filename.rsplit("/", 1)[-1].replace("\\", "_")
                    if not filename.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg")
                    ):
                        filename += ".jpg"
                    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)

                    try:
                        image_bytes = fetch_external_image(
                            image_url,
                            item.get("thumbnail", ""),
                            item.get("source_url", ""),
                        )
                        st.download_button(
                            label="⬇ Download Image",
                            data=image_bytes,
                            file_name=filename,
                            mime="application/octet-stream",
                            key=f"external_download_{start}_{offset}_{hash(image_url)}",
                            use_container_width=True,
                        )
                    except (requests.RequestException, ValueError) as exc:
                        st.caption("Download unavailable right now.")

                    if item.get("source_url"):
                        st.caption(
                            "Source and full licence details are available "
                            f"on the [image page]({item['source_url']})."
                        )

def load_theme_settings():
    """Load saved custom theme colors, falling back safely."""
    defaults = {
        "bg": "#F4EFE6",
        "surface": "#FFFCF6",
        "accent": "#8B6F47",
        "text": "#302B25",
        "muted": "#756B5F",
        "border": "#E7DED0",
        "button_text": "#FFFFFF",
    }

    try:
        with THEME_SETTINGS_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        for key in defaults:
            value = saved.get(key, defaults[key])
            if (
                isinstance(value, str)
                and len(value) == 7
                and value.startswith("#")
                and all(c in "0123456789abcdefABCDEF" for c in value[1:])
            ):
                defaults[key] = value
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    return defaults


def save_theme_settings(settings):
    """Persist custom theme colors to a local JSON file."""
    THEME_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with THEME_SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)


# --------------------------------------------------
# Themes
# --------------------------------------------------

THEMES = {
    "Midnight Charcoal": {
        "bg": "#111318",
        "surface": "#1B1E26",
        "accent": "#8B9CF6",
        "text": "#F3F4F6",
        "muted": "#A1A6B5",
        "border": "#303440",
        "button_text": "#111318",
    },
    "OLED Black": {
        "bg": "#000000",
        "surface": "#121212",
        "accent": "#A78BFA",
        "text": "#F5F5F5",
        "muted": "#AAAAAA",
        "border": "#292929",
        "button_text": "#111111",
    },
    "Emerald Dark": {
        "bg": "#0B1512",
        "surface": "#14231D",
        "accent": "#34D399",
        "text": "#E7F7EF",
        "muted": "#9DB9AC",
        "border": "#294438",
        "button_text": "#0B1512",
    },
    "Deep Ocean": {
        "bg": "#0A1424",
        "surface": "#14253A",
        "accent": "#38BDF8",
        "text": "#EAF5FF",
        "muted": "#A4BBD2",
        "border": "#263F5B",
        "button_text": "#0A1424",
    },
    "Violet Night": {
        "bg": "#151020",
        "surface": "#241A35",
        "accent": "#C084FC",
        "text": "#F5EEFF",
        "muted": "#B8A9CC",
        "border": "#413052",
        "button_text": "#151020",
    },
    "Clean Light": {
        "bg": "#F4EFE6",
        "surface": "#FFFCF6",
        "accent": "#8B6F47",
        "text": "#302B25",
        "muted": "#756B5F",
        "border": "#E7DED0",
        "button_text": "#FFFFFF",
    },
}


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="VectorForge | Image Search",
    page_icon="🔎",
    layout="wide",
)

with st.sidebar:
    st.markdown("## ◈ VectorForge")
    st.caption("AI-powered visual discovery")
    st.divider()

    theme_options = list(THEMES.keys()) + ["Custom Theme"]

    if "selected_theme" not in st.session_state:
        try:
            with THEME_SETTINGS_FILE.open("r", encoding="utf-8") as file:
                saved_settings = json.load(file)
            saved_theme = saved_settings.get("selected_theme", "Midnight Charcoal")
        except (OSError, json.JSONDecodeError):
            saved_theme = "Midnight Charcoal"

        st.session_state.selected_theme = (
            saved_theme if saved_theme in theme_options else "Midnight Charcoal"
        )

    def remember_theme_selection():
        st.session_state.selected_theme = st.session_state.theme_selector
        try:
            saved = load_theme_settings()
            saved["selected_theme"] = st.session_state.theme_selector
            save_theme_settings(saved)
        except OSError:
            pass

    theme_name = st.selectbox(
        "Appearance",
        theme_options,
        index=theme_options.index(st.session_state.selected_theme),
        key="theme_selector",
        on_change=remember_theme_selection,
    )
    st.session_state.selected_theme = theme_name

    if "custom_theme_colors" not in st.session_state:
        st.session_state.custom_theme_colors = load_theme_settings()

    if theme_name == "Custom Theme":
        st.markdown("### Customize colors")
        color_labels = {
            "bg": "Background",
            "surface": "Cards & surfaces",
            "accent": "Accent & buttons",
            "text": "Main text",
            "muted": "Muted text",
            "border": "Borders",
        }

        for color_key, label in color_labels.items():
            st.session_state.custom_theme_colors[color_key] = st.color_picker(
                label,
                value=st.session_state.custom_theme_colors[color_key],
                key=f"custom_color_{color_key}",
            )

        st.session_state.custom_theme_colors["button_text"] = (
            "#FFFFFF"
            if st.session_state.custom_theme_colors["accent"].lower()
            in {"#8b6f47", "#4f46e5", "#8b9cf6", "#a78bfa", "#c084fc", "#38bdf8", "#34d399"}
            else "#111111"
        )

        if st.button("Save custom theme", use_container_width=True):
            try:
                save_theme_settings(st.session_state.custom_theme_colors)
                st.success("Custom theme saved!")
            except OSError as exc:
                st.error(f"Could not save theme: {exc}")

        if st.button("Reset custom colors", use_container_width=True):
            defaults = {
                "bg": "#F4EFE6",
                "surface": "#FFFCF6",
                "accent": "#8B6F47",
                "text": "#302B25",
                "muted": "#756B5F",
                "border": "#E7DED0",
                "button_text": "#FFFFFF",
            }
            defaults["selected_theme"] = st.session_state.get(
                "selected_theme", "Custom Theme"
            )
            save_theme_settings(defaults)
            st.session_state.custom_theme_colors = {
                key: value for key, value in defaults.items()
                if key != "selected_theme"
            }
            for color_key in ("bg", "surface", "accent", "text", "muted", "border"):
                st.session_state.pop(f"custom_color_{color_key}", None)
            st.rerun()

theme = (
    st.session_state.custom_theme_colors
    if theme_name == "Custom Theme"
    else THEMES[theme_name]
)

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {"light" if theme_name == "Clean Light" else "dark"};
    }}

    .stApp {{
        background: {theme["bg"]};
        color: {theme["text"]};
    }}

    [data-testid="stHeader"] {{
        background: {theme["bg"]};
    }}

    [data-testid="stSidebar"] {{
        background: {theme["surface"]};
        border-right: 1px solid {theme["border"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {theme["text"]};
    }}

    /* Warm Beige: match the Appearance dropdown to the accent button */
    {"""
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background-color: #8B6F47 !important;
        border-color: #8B6F47 !important;
    }}

    [data-testid="stSelectbox"] [data-baseweb="select"] div,
    [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] input {{
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }}

    [data-testid="stSelectbox"] [data-baseweb="select"] svg {{
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }}

    [data-baseweb="popover"] [role="option"] {{
        color: #302B25 !important;
    }}
    """ if theme_name == "Clean Light" else ""}

    h1, h2, h3, h4, p, label {{
        color: {theme["text"]};
    }}

    [data-testid="stCaptionContainer"] p {{
        color: {theme["muted"]} !important;
    }}

    [data-testid="stTextInput"] input {{
        background: {theme["surface"]};
        color: {theme["text"]};
        border: 1px solid {theme["border"]};
        border-radius: 10px;
    }}

    [data-testid="stFileUploader"] section {{
        background: {theme["surface"]};
        border: 1px dashed {theme["border"]};
        border-radius: 12px;
    }}

    [data-testid="stRadio"] label {{
        color: {theme["text"]};
    }}

    .stButton button,
    .stDownloadButton button {{
        background: {theme["accent"]};
        color: {theme["button_text"]};
        border: 1px solid transparent;
        border-radius: 10px;
        font-weight: 600;
        min-height: 42px;
        transition: filter 0.15s ease;
    }}

    .stButton button:hover,
    .stDownloadButton button:hover {{
        filter: brightness(1.08);
        border: 1px solid {theme["accent"]};
    }}

    [data-testid="stMetric"] {{
        background: {theme["surface"]};
        border: 1px solid {theme["border"]};
        padding: 14px;
        border-radius: 12px;
    }}

    [data-testid="stImage"] {{
        overflow: hidden;
        border-radius: 10px;
    }}

    hr {{
        border-color: {theme["border"]};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.caption("AI · VECTOR SEARCH · CLIP")
st.title("Find images by meaning.")
st.write(
    "Discover similar images using natural language "
    "or a visual reference."
)
st.divider()


# Model and mapping
# --------------------------------------------------

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
        vector = vector / vector.norm(
            dim=-1,
            keepdim=True,
        )

    return vector.squeeze(0).numpy()


def embed_image(image, model, preprocess):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vector = model.encode_image(tensor)
        vector = vector / vector.norm(
            dim=-1,
            keepdim=True,
        )

    return vector.squeeze(0).numpy()


def get_image_path(key):
    if key.startswith("coco/"):
        return (
            PROJECT
            / "datasets"
            / "coco"
            / "images"
            / key.split("/", 1)[1]
        )

    return PROJECT / "images" / key


# --------------------------------------------------
# Premium results grid
# --------------------------------------------------

def show_results(results, mapping):
    id_to_key = {
        int(image_id): key
        for key, image_id in mapping.items()
    }

    visible_results = []

    for result in results:
        image_id = int(result[0])
        distance = float(result[1])
        key = id_to_key.get(image_id)

        if key is None:
            continue

        image_path = get_image_path(key)

        if image_path.is_file():
            visible_results.append(
                (image_id, image_path, distance)
            )

    if not visible_results:
        st.info(
            "No displayable local images found. "
            "Try another query."
        )
        return

    st.markdown(
        f"### Search results"
        f" <span style='color:{theme['muted']};"
        f"font-size:15px'>({len(visible_results)} images)</span>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Ranked by vector similarity · Lower cosine distance "
        "means a closer match."
    )

    columns_per_row = 4

    for start_index in range(
        0,
        len(visible_results),
        columns_per_row,
    ):
        row_results = visible_results[
            start_index:start_index + columns_per_row
        ]
        columns = st.columns(columns_per_row)

        for offset, item in enumerate(row_results):
            image_id, image_path, distance = item
            rank = start_index + offset + 1

            with columns[offset]:
                with st.container(
                    border=True,
                    key=f"result_card_{image_id}",
                ):
                    st.markdown(
                        f"<div style='color:{theme['accent']};"
                        f"font-size:12px;font-weight:700;"
                        f"letter-spacing:1px;margin-bottom:8px'>"
                        f"RESULT #{rank:02d}</div>",
                        unsafe_allow_html=True,
                    )

                    st.image(
                        str(image_path),
                        width="stretch",
                    )

                    st.markdown(
                        f"**{image_path.name}**"
                    )

                    st.markdown(
                        f"<div style='color:{theme['muted']};"
                        f"font-size:13px;margin:8px 0 12px 0'>"
                        f"Cosine distance: "
                        f"<strong style='color:{theme['text']}'>"
                        f"{distance:.4f}</strong></div>",
                        unsafe_allow_html=True,
                    )

                    mime_type, _ = mimetypes.guess_type(
                        str(image_path)
                    )
                    mime_type = (
                        mime_type or "application/octet-stream"
                    )

                    st.download_button(
                        label="⬇ Download Image",
                        data=image_path.read_bytes(),
                        file_name=image_path.name,
                        mime=mime_type,
                        key=f"download_{image_id}",
                        use_container_width=True,
                    )


# Initialize app
# --------------------------------------------------

try:
    mapping = load_mapping()
    model, preprocess, tokenizer = load_model()

except Exception as exc:
    st.error(
        f"Could not initialize VectorForge: {exc}"
    )
    st.stop()


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------

with st.sidebar:
    st.divider()
    st.markdown("### Your library")

    st.metric(
        "Indexed images",
        len(mapping),
    )

    result_count = st.slider(
        "Results to display",
        min_value=1,
        max_value=20,
        value=8,
    )

    st.caption(
        "Lower cosine distance = closer match."
    )


# --------------------------------------------------
# Search controls
# --------------------------------------------------

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "search_label" not in st.session_state:
    st.session_state.search_label = ""

if "external_results" not in st.session_state:
    st.session_state.external_results = None

if "external_query" not in st.session_state:
    st.session_state.external_query = ""

def clear_results_on_query_change():
    """Clear old results as soon as the text query changes."""
    st.session_state.search_results = None
    st.session_state.search_label = ""
    st.session_state.external_results = None
    st.session_state.external_query = ""


mode = st.radio(
    "Search method",
    ["Text Search", "Image Search"],
    horizontal=True,
)

if mode == "Text Search":
    st.markdown(
        "### Describe what you're looking for"
    )

    query = st.text_input(
        "Text query",
        placeholder="e.g. a cat sitting on a sofa",
        label_visibility="collapsed",
        key="text_query",
        on_change=clear_results_on_query_change,
    )

    st.caption(
        "Examples: a cat · a red car · a dog in a park"
    )


    if st.button(
        "⌕ Search images",
        type="primary",
    ):
        if not query.strip():
            st.warning("Enter a description first.")

        else:
            # Reset previous results before starting a new search.
            st.session_state.search_results = None
            st.session_state.search_label = ""
            st.session_state.external_results = None
            st.session_state.external_query = ""

            try:
                with st.spinner("Preparing your search..."):
                    vector = embed_text(
                        query.strip(),
                        model,
                        tokenizer,
                    )
            except Exception as exc:
                st.error(f"Could not prepare search: {exc}")
            else:
                local_results = []
                local_error = None

                try:
                    with st.spinner("Searching your image library..."):
                        with vf.Collection(
                            dim=512,
                            metric="cosine",
                            path=str(DB_DIR),
                        ) as db:
                            local_results = db.search(
                                vector,
                                k=result_count,
                            )
                except Exception as exc:
                    local_error = exc

                # Keep only local matches below the relevance cutoff.
                local_distance_threshold = 0.72
                relevant_local_results = [
                    result
                    for result in local_results
                    if len(result) > 1
                    and float(result[1]) <= local_distance_threshold
                ]

                st.session_state.search_results = relevant_local_results
                st.session_state.search_label = (
                    f"Local matches: {query.strip()}"
                )

                # Fill remaining slots from Wikimedia, even if the
                # local database is unavailable in the cloud.
                remaining_slots = max(
                    0, result_count - len(relevant_local_results)
                )

                if remaining_slots > 0:
                    try:
                        with st.spinner(
                            "Finding additional relevant images..."
                        ):
                            st.session_state.external_results = (
                                search_wikimedia_images(
                                    query.strip(),
                                    limit=remaining_slots,
                                )
                            )
                            st.session_state.external_query = query.strip()
                    except Exception as exc:
                        st.warning(f"External image search failed: {exc}")

                if local_error is not None:
                    st.caption(
                        "Your local image library is unavailable here; "
                        "showing external results where available."
                    )

else:
    st.markdown(
        "### Find visually similar images"
    )

    st.caption(
        "Upload a reference image to find visually "
        "related images in your collection."
    )

    uploaded_file = st.file_uploader(
        "Choose a reference image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                uploaded_image,
                caption="Your reference image",
                width=280,
            )

            if st.button(
                "⌕ Find similar images",
                type="primary",
            ):
                with st.spinner(
                    "Comparing visual embeddings..."
                ):
                    vector = embed_image(
                        uploaded_image,
                        model,
                        preprocess,
                    )

                    with vf.Collection(
                        dim=512,
                        metric="cosine",
                        path=str(DB_DIR),
                    ) as db:
                        results = db.search(
                            vector,
                            k=result_count,
                        )

                st.session_state.search_results = results
                st.session_state.search_label = (
                    "Visually similar images"
                )

        except Exception as exc:
            st.error(
                f"Could not process image: {exc}"
            )

if st.session_state.search_results is not None:
    st.markdown(
        f"#### {st.session_state.search_label}"
    )
    show_results(
        st.session_state.search_results,
        mapping,
    )


if st.session_state.external_results is not None:
    st.caption(
        f"External search query: {st.session_state.external_query}"
    )
    show_external_results(st.session_state.external_results)


# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "VECTORFORGE  /  CLIP embeddings  /  "
    "Vector similarity search"
)
