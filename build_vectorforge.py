from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "VectorForge"
BUILD_DIR = SOURCE_DIR / "build"
LIBRARY = BUILD_DIR / "libvectorforge.so"


def ensure_vectorforge_built():
    if LIBRARY.exists():
        return LIBRARY

    subprocess.run(
        [
            "cmake",
            "-S", str(SOURCE_DIR),
            "-B", str(BUILD_DIR),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DVF_NATIVE=OFF",
            "-DVF_BUILD_TESTS=OFF",
        ],
        check=True,
    )

    subprocess.run(
        [
            "cmake",
            "--build", str(BUILD_DIR),
            "--target", "vectorforge_shared",
            "-j2",
        ],
        check=True,
    )

    if not LIBRARY.exists():
        raise FileNotFoundError(
            f"VectorForge shared library was not created: {LIBRARY}"
        )

    return LIBRARY
