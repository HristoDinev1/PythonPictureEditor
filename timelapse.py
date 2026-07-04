import shutil
import subprocess
from pathlib import Path

from PIL import Image

from image_processor import FRAME_NAME_TEMPLATE


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


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


def export_gif(frame_paths: list[Path], output_path: Path, fps: int = 24) -> Path:
    if not frame_paths:
        raise ValueError("cannot export a timelapse from zero frames")
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    first, *rest = frames
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=max(1, round(1000 / fps)),
        loop=0,
    )
    return output_path


def export_timelapse(
    frame_paths: list[Path],
    output_directory: Path,
    fps: int = 24,
) -> Path:
    if not frame_paths:
        raise ValueError("cannot export a timelapse from zero frames")
    output_directory.mkdir(parents=True, exist_ok=True)
    if ffmpeg_available():
        return export_mp4(
            frame_paths[0].parent, output_directory / "timelapse.mp4", fps
        )
    return export_gif(frame_paths, output_directory / "timelapse.gif", fps)
