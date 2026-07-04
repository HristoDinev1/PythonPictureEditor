from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

import utils
from main import Settings, apply_settings
from film_simulation import FilmPreset

FRAME_NAME_TEMPLATE: str = "frame_{index:04d}.png"


def process_images(
    images: list[np.ndarray],
    settings: Settings,
    preset: FilmPreset | None = None,
    max_workers: int | None = None,
) -> list[np.ndarray]:
    if not images:
        return []
    worker = partial(apply_settings, settings=settings, preset=preset)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, images))


def process_images_serial(
    images: list[np.ndarray],
    settings: Settings,
    preset: FilmPreset | None = None,
) -> list[np.ndarray]:
    return [apply_settings(image, settings, preset) for image in images]


def _render_frame(
    indexed_path: tuple[int, Path],
    settings: Settings,
    preset: FilmPreset | None,
    output_directory: Path,
) -> Path:
    index, source_path = indexed_path
    image = utils.load_image(source_path)
    edited = apply_settings(image, settings, preset)
    destination = output_directory / FRAME_NAME_TEMPLATE.format(index=index)
    return utils.save_image(edited, destination)


def process_files(
    paths: list[Path],
    settings: Settings,
    output_directory: Path,
    preset: FilmPreset | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    if not paths:
        return []
    output_directory.mkdir(parents=True, exist_ok=True)
    worker = partial(
        _render_frame,
        settings=settings,
        preset=preset,
        output_directory=output_directory,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, enumerate(paths)))
