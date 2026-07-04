import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from main import (
    FilmPreset,
    Settings,
    add_settings_arguments,
    apply_settings,
    list_photo_files,
    load_image,
    save_image,
    settings_from_args,
)

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
    image = load_image(source_path)
    edited = apply_settings(image, settings, preset)
    destination = output_directory / FRAME_NAME_TEMPLATE.format(index=index)
    return save_image(edited, destination)


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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=None)
    add_settings_arguments(parser)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    settings = settings_from_args(args)
    paths = list_photo_files(args.source_dir)
    frames = process_files(paths, settings, args.output_dir, max_workers=args.workers)
    print(f"Wrote {len(frames)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
