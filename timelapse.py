import shutil
import subprocess
from pathlib import Path

import numpy as np

import image_processor
import utils
from image_processor import FRAME_NAME_TEMPLATE, ProgressCallback
from film_simulation import FilmPreset
from main import Settings

MODE_STANDARD: str = "standard"
MODE_HYPERLAPSE: str = "hyperlapse"
MODE_LONG_EXPOSURE: str = "long_exposure"
DEFAULT_LONG_EXPOSURE_WINDOW: int = 10


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def estimate_duration_seconds(
    photo_count: int,
    mode: str,
    fps: int = 24,
    long_exposure_window: int = DEFAULT_LONG_EXPOSURE_WINDOW,
) -> float:
    if photo_count <= 0:
        return 0.0
    if mode == MODE_LONG_EXPOSURE:
        frame_count = -(-photo_count // max(1, long_exposure_window))
    else:
        frame_count = photo_count
    return frame_count / fps


def blend_long_exposure(
    frame_paths: list[Path],
    output_directory: Path,
    window: int = DEFAULT_LONG_EXPOSURE_WINDOW,
) -> list[Path]:
    if not frame_paths:
        raise ValueError("cannot build a long-exposure timelapse from zero frames")
    output_directory.mkdir(parents=True, exist_ok=True)
    window = max(1, window)
    blended_paths = []
    for output_index, start in enumerate(range(0, len(frame_paths), window)):
        chunk = frame_paths[start : start + window]
        blended = utils.load_image(chunk[0])
        for path in chunk[1:]:
            blended = np.maximum(blended, utils.load_image(path))
        destination = output_directory / FRAME_NAME_TEMPLATE.format(index=output_index)
        blended_paths.append(utils.save_image(blended, destination))
    return blended_paths


def _estimate_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int]:
    freq_reference = np.fft.fft2(reference)
    freq_moving = np.fft.fft2(moving)
    cross_power = freq_reference * np.conj(freq_moving)
    magnitude = np.maximum(np.abs(cross_power), 1e-8)
    correlation = np.fft.ifft2(cross_power / magnitude).real
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    height, width = correlation.shape
    if peak_y > height // 2:
        peak_y -= height
    if peak_x > width // 2:
        peak_x -= width
    return int(peak_y), int(peak_x)


def stabilize_hyperlapse(frame_paths: list[Path], output_directory: Path) -> list[Path]:
    if not frame_paths:
        raise ValueError("cannot build a hyperlapse from zero frames")
    output_directory.mkdir(parents=True, exist_ok=True)
    images = [utils.load_image(path) for path in frame_paths]
    if len(images) == 1:
        return [
            utils.save_image(
                images[0], output_directory / FRAME_NAME_TEMPLATE.format(index=0)
            )
        ]

    grays = [image.mean(axis=-1) for image in images]
    shifts = [(0, 0)]
    cumulative_y = cumulative_x = 0
    for previous, current in zip(grays, grays[1:]):
        dy, dx = _estimate_shift(previous, current)
        cumulative_y += dy
        cumulative_x += dx
        shifts.append((cumulative_y, cumulative_x))

    height, width = grays[0].shape
    top = max(0, max(y for y, _ in shifts))
    bottom = max(0, -min(y for y, _ in shifts))
    left = max(0, max(x for _, x in shifts))
    right = max(0, -min(x for _, x in shifts))
    crop_top, crop_bottom = top, height - bottom
    crop_left, crop_right = left, width - right
    if crop_bottom <= crop_top or crop_right <= crop_left:
        shifts = [(0, 0)] * len(images)
        crop_top, crop_bottom, crop_left, crop_right = 0, height, 0, width

    aligned_paths = []
    for index, (image, (shift_y, shift_x)) in enumerate(zip(images, shifts)):
        shifted = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))
        cropped = shifted[crop_top:crop_bottom, crop_left:crop_right]
        destination = output_directory / FRAME_NAME_TEMPLATE.format(index=index)
        aligned_paths.append(utils.save_image(cropped, destination))
    return aligned_paths


def prepare_frames(
    frame_paths: list[Path],
    mode: str,
    output_directory: Path,
    long_exposure_window: int = DEFAULT_LONG_EXPOSURE_WINDOW,
) -> list[Path]:
    if mode == MODE_LONG_EXPOSURE:
        return blend_long_exposure(frame_paths, output_directory, long_exposure_window)
    if mode == MODE_HYPERLAPSE:
        return stabilize_hyperlapse(frame_paths, output_directory)
    return frame_paths


def export_mp4(frames_directory: Path, output_path: Path, fps: int = 24) -> Path:
    frame_pattern = frames_directory / FRAME_NAME_TEMPLATE.replace(
        "{index:04d}", "%04d"
    )
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def export_timelapse(
    frame_paths: list[Path],
    output_directory: Path,
    fps: int = 24,
) -> Path:
    if not frame_paths:
        raise ValueError("cannot export a timelapse from zero frames")
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to export a timelapse but was not found on PATH"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    return export_mp4(frame_paths[0].parent, output_directory / "timelapse.mp4", fps)


def build_timelapse(
    photo_paths: list[Path],
    output_directory: Path,
    settings: Settings | None = None,
    preset: FilmPreset | None = None,
    mode: str = MODE_STANDARD,
    fps: int = 24,
    long_exposure_window: int = DEFAULT_LONG_EXPOSURE_WINDOW,
    work_directory: Path | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    # Build a timelapse straight from arbitrary photos: no separate "apply to all
    # photos" step is required first. Each photo is edited with the current
    # settings (a no-op if nothing is dialled in), resized to a common size so
    # ffmpeg gets uniform frames, written as a sequential frame, then handed to
    # the chosen timelapse mode and encoded.
    if not photo_paths:
        raise ValueError("cannot build a timelapse from zero photos")
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to export a timelapse but was not found on PATH"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    work = work_directory or (output_directory / "_timelapse_work")
    first = utils.load_image(photo_paths[0])
    target_size = (first.shape[0], first.shape[1])
    frames = image_processor.process_files(
        photo_paths,
        settings if settings is not None else Settings(),
        work / "frames",
        preset,
        target_size=target_size,
        progress=progress,
    )
    prepared = prepare_frames(
        frames, mode, work / "prepared", long_exposure_window
    )
    return export_timelapse(prepared, output_directory, fps)
