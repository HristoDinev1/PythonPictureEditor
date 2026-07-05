import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import image_processor
import main
import timelapse
import utils
from main import Settings

FIXTURES = Path(__file__).parent / "fixtures"

# Every filter (slider) the editor exposes, as (label, function).
FILTERS: list[tuple[str, object]] = [
    ("Exposure", main.adjust_exposure),
    ("Brightness", main.adjust_brightness),
    ("Contrast", main.adjust_contrast),
    ("Highlights", main.adjust_highlights),
    ("Shadows", main.adjust_shadows),
    ("Whites", main.adjust_whites),
    ("Blacks", main.adjust_blacks),
    ("Temperature", main.adjust_temperature),
    ("Tint", main.adjust_tint),
    ("Vibrance", main.adjust_vibrance),
    ("Saturation", main.adjust_saturation),
    ("Texture", main.adjust_texture),
    ("Clarity", main.adjust_clarity),
    ("Dehaze", main.adjust_dehaze),
    ("Milky Way Glow", main.adjust_milky_way_glow),
    ("Vignette", main.adjust_vignette),
    ("Sharpness", main.adjust_sharpness),
    ("Noise Reduction", main.reduce_luminance_noise),
    ("Noise Reduction (Color)", main.reduce_color_noise),
]

MINIMUM = -1.0
MAXIMUM = 1.0


def make_sample_image(height: int = 120, width: int = 160) -> np.ndarray:
    # A colourful gradient kept in the low-midtones so no single filter pushes the
    # whole image out of another filter's range -- every filter has content to act on.
    rows = np.linspace(0.08, 0.42, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.10, 0.40, width, dtype=np.float32)[None, :]
    red = rows * np.ones_like(cols)
    green = np.ones_like(rows) * cols
    blue = (rows + cols) / 2.0
    return np.stack([red, green, blue], axis=-1)


def filter_changes_image(filter_function, image: np.ndarray) -> bool:
    for amount in (MINIMUM, MAXIMUM):
        if not np.allclose(filter_function(image, amount), image):
            return True
    return False


class TestEveryFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_sample_image()

    def test_each_filter_changes_the_image_at_minimum_and_maximum(self) -> None:
        for name, filter_function in FILTERS:
            with self.subTest(filter=name):
                minimum_result = filter_function(self.image, MINIMUM)
                maximum_result = filter_function(self.image, MAXIMUM)
                self.assertFalse(
                    np.allclose(minimum_result, self.image),
                    f"{name} at minimum did nothing",
                )
                self.assertFalse(
                    np.allclose(maximum_result, self.image),
                    f"{name} at maximum did nothing",
                )

    def test_the_other_filters_still_work_for_every_filter(self) -> None:
        for name, filter_function in FILTERS:
            with self.subTest(focused=name):
                for other_name, other_filter in FILTERS:
                    if other_filter is filter_function:
                        continue
                    self.assertTrue(
                        filter_changes_image(other_filter, self.image),
                        f"{other_name} stopped working alongside {name}",
                    )


class TestLoadAndSaveImage(unittest.TestCase):
    def test_can_load_an_image_and_then_save_it(self) -> None:
        source = sorted(FIXTURES.glob("*.jpg"))[0]
        image = utils.load_image(source)
        with tempfile.TemporaryDirectory() as temp:
            destination = utils.save_image(image, Path(temp) / "saved.png")
            self.assertTrue(destination.exists())
            reloaded = utils.load_image(destination)
            self.assertEqual(reloaded.shape, image.shape)


class TestFolderToFolder(unittest.TestCase):
    def test_open_a_folder_and_save_everything_to_another_location(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            input_directory = temp_path / "input"
            output_directory = temp_path / "output"
            input_directory.mkdir()

            fixtures = sorted(FIXTURES.glob("*.jpg"))
            self.assertEqual(len(fixtures), 3)
            for fixture in fixtures:
                shutil.copy(fixture, input_directory / fixture.name)

            paths = utils.list_photo_files(input_directory)
            self.assertEqual(len(paths), 3)

            saved = image_processor.process_files(paths, Settings(), output_directory)
            self.assertEqual(len(saved), 3)
            for path in saved:
                self.assertTrue(path.exists())
            self.assertEqual(len(list(output_directory.glob("frame_*.png"))), 3)


class TestTimelapseExportIsAnOption(unittest.TestCase):
    def test_ffmpeg_availability_is_reported_as_a_boolean(self) -> None:
        self.assertIsInstance(timelapse.ffmpeg_available(), bool)

    def test_a_timelapse_can_be_exported_from_photos(self) -> None:
        photos = sorted(FIXTURES.glob("*.jpg"))
        with tempfile.TemporaryDirectory() as temp:
            with (
                mock.patch("timelapse.ffmpeg_available", return_value=True),
                mock.patch("timelapse.subprocess.run") as mock_run,
            ):
                output = timelapse.build_timelapse(photos, Path(temp))
            self.assertEqual(output.suffix, ".mp4")
            mock_run.assert_called_once()

    def test_timelapse_does_not_require_pre_applied_filters(self) -> None:
        photos = sorted(FIXTURES.glob("*.jpg"))
        with tempfile.TemporaryDirectory() as temp:
            with (
                mock.patch("timelapse.ffmpeg_available", return_value=True),
                mock.patch("timelapse.subprocess.run"),
            ):
                output = timelapse.build_timelapse(photos, Path(temp))
            self.assertEqual(output.suffix, ".mp4")

    def test_missing_ffmpeg_reports_a_clear_error(self) -> None:
        photos = sorted(FIXTURES.glob("*.jpg"))
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch("timelapse.ffmpeg_available", return_value=False):
                with self.assertRaises(RuntimeError):
                    timelapse.build_timelapse(photos, Path(temp))


if __name__ == "__main__":
    unittest.main()
