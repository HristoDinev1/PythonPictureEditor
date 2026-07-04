from pathlib import Path

import numpy as np
from PIL import Image

STANDARD_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
RAW_EXTENSIONS: frozenset[str] = frozenset(
    {".cr2", ".cr3", ".craw", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
)
SUPPORTED_EXTENSIONS: frozenset[str] = STANDARD_EXTENSIONS | RAW_EXTENSIONS


def is_raw_file(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def list_photo_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _load_raw_image(path: Path) -> np.ndarray:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess()
    return rgb.astype(np.float32) / 255.0


def _load_standard_image(path: Path) -> np.ndarray:
    with Image.open(path) as picture:
        rgb = picture.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def load_image(path: Path) -> np.ndarray:
    if is_raw_file(path):
        return _load_raw_image(path)
    return _load_standard_image(path)


def to_pil_image(image: np.ndarray) -> Image.Image:
    eight_bit = (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(eight_bit, mode="RGB")


def save_image(image: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(image).save(path)
    return path
