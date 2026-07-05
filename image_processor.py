from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np

import utils
from main import Settings, apply_settings
from film_simulation import FilmPreset

FRAME_NAME_TEMPLATE: str = "frame_{index:04d}.png"
ProgressCallback = Callable[[int, int], None]


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
    target_size: tuple[int, int] | None,
) -> Path:
    index, source_path = indexed_path
    image = utils.load_image(source_path)
    edited = apply_settings(image, settings, preset)
    if target_size is not None:
        edited = utils.resize_image(edited, target_size[0], target_size[1])
    destination = output_directory / FRAME_NAME_TEMPLATE.format(index=index)
    return utils.save_image(edited, destination)


def process_files(
    paths: list[Path],
    settings: Settings,
    output_directory: Path,
    preset: FilmPreset | None = None,
    max_workers: int | None = None,
    target_size: tuple[int, int] | None = None,
    progress: ProgressCallback | None = None,
) -> list[Path]:
    if not paths:
        return []
    output_directory.mkdir(parents=True, exist_ok=True)
    worker = partial(
        _render_frame,
        settings=settings,
        preset=preset,
        output_directory=output_directory,
        target_size=target_size,
    )
    results: list[Path] = [Path()] * len(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(worker, (index, path)): index
            for index, path in enumerate(paths)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            if progress is not None:
                progress(completed, len(paths))
    return results
