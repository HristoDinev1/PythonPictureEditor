import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

LUMA_WEIGHTS: np.ndarray = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

STANDARD_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
RAW_EXTENSIONS: frozenset[str] = frozenset(
    {".cr2", ".cr3", ".craw", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
)
SUPPORTED_EXTENSIONS: frozenset[str] = STANDARD_EXTENSIONS | RAW_EXTENSIONS


@dataclass(frozen=True)
class FilmPreset:
    name: str
    tone_curve: list[list[float]]
    color_matrix: list[list[float]]
    grain_amount: float
    grain_seed: int


@dataclass(frozen=True)
class Settings:
    exposure: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    vignette: float = 0.0
    noise_reduction_luminance: float = 0.0
    noise_reduction_color: float = 0.0
    film_simulation: str = field(default="None")


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


def clip_unit(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def luminance(image: np.ndarray) -> np.ndarray:
    return image @ LUMA_WEIGHTS


def smoothstep(values: np.ndarray, edge_low: float, edge_high: float) -> np.ndarray:
    ramp = np.clip((values - edge_low) / (edge_high - edge_low), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _box_blur_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    window = 2 * radius + 1
    pad_width = [
        (radius, radius) if index == axis else (0, 0) for index in range(values.ndim)
    ]
    padded = np.pad(values, pad_width, mode="edge")
    zero_shape = list(padded.shape)
    zero_shape[axis] = 1
    running = np.concatenate(
        [
            np.zeros(zero_shape, dtype=np.float32),
            np.cumsum(padded, axis=axis, dtype=np.float32),
        ],
        axis=axis,
    )
    upper = np.take(running, range(window, running.shape[axis]), axis=axis)
    lower = np.take(running, range(0, running.shape[axis] - window), axis=axis)
    return ((upper - lower) / window).astype(np.float32)


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image
    blurred = _box_blur_axis(image, radius, axis=0)
    return _box_blur_axis(blurred, radius, axis=1)


def adjust_exposure(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return clip_unit(image * float(2.0 ** (2.0 * amount)))


def adjust_contrast(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return clip_unit((image - 0.5) * (1.0 + amount) + 0.5)


def _tonal_shift(image: np.ndarray, amount: float, weight: np.ndarray) -> np.ndarray:
    return clip_unit(image + 0.5 * amount * weight[..., None])


def adjust_highlights(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _tonal_shift(image, amount, smoothstep(luminance(image), 0.5, 1.0))


def adjust_shadows(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _tonal_shift(image, amount, 1.0 - smoothstep(luminance(image), 0.0, 0.5))


def adjust_whites(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _tonal_shift(image, amount, smoothstep(luminance(image), 0.75, 1.0))


def adjust_blacks(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _tonal_shift(image, amount, 1.0 - smoothstep(luminance(image), 0.0, 0.25))


def adjust_temperature(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    scale = np.array(
        [1.0 + 0.25 * amount, 1.0, 1.0 - 0.25 * amount], dtype=np.float32
    )
    return clip_unit(image * scale)


def adjust_tint(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    scale = np.array(
        [1.0 - 0.1 * amount, 1.0 + 0.2 * amount, 1.0 - 0.1 * amount],
        dtype=np.float32,
    )
    return clip_unit(image * scale)


def adjust_saturation(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    gray = luminance(image)[..., None]
    return clip_unit(gray + (image - gray) * (1.0 + amount))


def adjust_vibrance(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    saturation_level = image.max(axis=-1) - image.min(axis=-1)
    boost = 1.0 + amount * (1.0 - saturation_level)[..., None]
    gray = luminance(image)[..., None]
    return clip_unit(gray + (image - gray) * boost)


def _local_contrast(image: np.ndarray, amount: float, radius: int) -> np.ndarray:
    detail = image - box_blur(image, radius)
    return clip_unit(image + amount * detail)


def adjust_texture(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _local_contrast(image, 1.5 * amount, radius=2)


def adjust_clarity(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    radius = max(4, min(image.shape[0], image.shape[1]) // 50)
    return _local_contrast(image, amount, radius)


def tiled_equalize(gray: np.ndarray, tiles: int = 8, bins: int = 256) -> np.ndarray:
    height, width = gray.shape
    grid = min(tiles, height, width)
    levels = np.clip((gray * (bins - 1)).astype(np.int64), 0, bins - 1)
    row_edges = np.linspace(0, height, grid + 1).astype(np.int64)
    col_edges = np.linspace(0, width, grid + 1).astype(np.int64)
    lookup_tables = np.zeros((grid, grid, bins), dtype=np.float64)
    for row in range(grid):
        for col in range(grid):
            tile = levels[
                row_edges[row] : row_edges[row + 1],
                col_edges[col] : col_edges[col + 1],
            ]
            histogram = np.bincount(tile.ravel(), minlength=bins)
            cumulative = np.cumsum(histogram).astype(np.float64)
            lookup_tables[row, col] = cumulative / max(cumulative[-1], 1.0)
    tile_height = height / grid
    tile_width = width / grid
    row_positions = np.clip(
        (np.arange(height) + 0.5) / tile_height - 0.5, 0.0, grid - 1.0
    )
    col_positions = np.clip(
        (np.arange(width) + 0.5) / tile_width - 0.5, 0.0, grid - 1.0
    )
    row_low = row_positions.astype(np.int64)[:, None]
    col_low = col_positions.astype(np.int64)[None, :]
    row_high = np.minimum(row_low + 1, grid - 1)
    col_high = np.minimum(col_low + 1, grid - 1)
    row_frac = (row_positions[:, None] - row_low).astype(np.float64)
    col_frac = (col_positions[None, :] - col_low).astype(np.float64)
    top = (1.0 - col_frac) * lookup_tables[row_low, col_low, levels] + (
        col_frac * lookup_tables[row_low, col_high, levels]
    )
    bottom = (1.0 - col_frac) * lookup_tables[row_high, col_low, levels] + (
        col_frac * lookup_tables[row_high, col_high, levels]
    )
    return ((1.0 - row_frac) * top + row_frac * bottom).astype(np.float32)


def adjust_dehaze(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    gray = luminance(image)
    target = tiled_equalize(gray) if amount > 0.0 else box_blur(gray, radius=8)
    blended = gray + abs(amount) * (target - gray)
    ratio = blended / np.maximum(gray, 1e-6)
    return clip_unit(image * ratio[..., None])


def adjust_vignette(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    height, width = image.shape[:2]
    rows = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    distance = np.sqrt(rows * rows + cols * cols) / np.sqrt(2.0)
    falloff = smoothstep(distance, 0.3, 1.0)
    return clip_unit(image * (1.0 - amount * falloff)[..., None])


def reduce_luminance_noise(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    gray = luminance(image)
    smoothed = gray + amount * (box_blur(gray, radius=2) - gray)
    ratio = smoothed / np.maximum(gray, 1e-6)
    return clip_unit(image * ratio[..., None])


def reduce_color_noise(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    gray = luminance(image)[..., None]
    chroma = image - gray
    smoothed_chroma = chroma + amount * (box_blur(chroma, radius=3) - chroma)
    return clip_unit(gray + smoothed_chroma)


def apply_tone_curve(image: np.ndarray, curve: list[list[float]]) -> np.ndarray:
    points = np.array(curve, dtype=np.float32)
    return np.interp(image, points[:, 0], points[:, 1]).astype(np.float32)


def apply_color_matrix(image: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    return clip_unit(image @ np.array(matrix, dtype=np.float32).T)


def apply_grain(image: np.ndarray, amount: float, seed: int) -> np.ndarray:
    if amount == 0.0:
        return image
    generator = np.random.default_rng(seed)
    noise = generator.normal(0.0, amount, size=image.shape[:2]).astype(np.float32)
    return clip_unit(image + noise[..., None])


def apply_film_preset(image: np.ndarray, preset: FilmPreset) -> np.ndarray:
    curved = apply_tone_curve(image, preset.tone_curve)
    graded = apply_color_matrix(curved, preset.color_matrix)
    return apply_grain(graded, preset.grain_amount, preset.grain_seed)


def apply_settings(
    image: np.ndarray,
    settings: Settings,
    preset: FilmPreset | None = None,
) -> np.ndarray:
    scaled = {
        name: getattr(settings, name) / 100.0
        for name in (
            "exposure",
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "temperature",
            "tint",
            "vibrance",
            "saturation",
            "texture",
            "clarity",
            "dehaze",
            "vignette",
            "noise_reduction_luminance",
            "noise_reduction_color",
        )
    }
    result = image.astype(np.float32)
    result = adjust_exposure(result, scaled["exposure"])
    result = adjust_temperature(result, scaled["temperature"])
    result = adjust_tint(result, scaled["tint"])
    result = adjust_contrast(result, scaled["contrast"])
    result = adjust_highlights(result, scaled["highlights"])
    result = adjust_shadows(result, scaled["shadows"])
    result = adjust_whites(result, scaled["whites"])
    result = adjust_blacks(result, scaled["blacks"])
    result = adjust_vibrance(result, scaled["vibrance"])
    result = adjust_saturation(result, scaled["saturation"])
    result = adjust_texture(result, scaled["texture"])
    result = adjust_clarity(result, scaled["clarity"])
    result = adjust_dehaze(result, scaled["dehaze"])
    result = reduce_luminance_noise(result, scaled["noise_reduction_luminance"])
    result = reduce_color_noise(result, scaled["noise_reduction_color"])
    result = adjust_vignette(result, scaled["vignette"])
    if preset is not None:
        result = apply_film_preset(result, preset)
    return result


SETTINGS_CLI_FIELDS: tuple[str, ...] = (
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "temperature",
    "tint",
    "vibrance",
    "saturation",
    "texture",
    "clarity",
    "dehaze",
    "vignette",
    "noise_reduction_luminance",
    "noise_reduction_color",
)


def add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    for field_name in SETTINGS_CLI_FIELDS:
        parser.add_argument(
            f"--{field_name.replace('_', '-')}", type=float, default=0.0
        )


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(**{name: getattr(args, name) for name in SETTINGS_CLI_FIELDS})


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    add_settings_arguments(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    settings = settings_from_args(args)
    image = load_image(args.input_path)
    result = apply_settings(image, settings)
    saved_path = save_image(result, args.output_path)
    print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()
