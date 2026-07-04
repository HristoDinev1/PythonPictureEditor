import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import utils


class TestFolderListing(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_empty_folder_returns_empty_list(self) -> None:
        self.assertEqual(utils.list_photo_files(self.directory), [])

    def test_only_supported_extensions_are_listed(self) -> None:
        for name in ("a.jpg", "b.PNG", "c.cr3", "notes.txt", "d.tiff"):
            (self.directory / name).touch()
        listed = [path.name for path in utils.list_photo_files(self.directory)]
        self.assertEqual(listed, ["a.jpg", "b.PNG", "c.cr3"])

    def test_listing_is_sorted(self) -> None:
        for name in ("c.png", "a.png", "b.png"):
            (self.directory / name).touch()
        listed = [path.name for path in utils.list_photo_files(self.directory)]
        self.assertEqual(listed, sorted(listed))

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(NotADirectoryError):
            utils.list_photo_files(self.directory / "does_not_exist")


class TestImageRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        gradient = np.linspace(0.0, 1.0, 32, dtype=np.float32)
        self.image = np.stack(
            [np.tile(gradient, (16, 1))] * 3,
            axis=-1,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_png_round_trip_preserves_pixels(self) -> None:
        path = self.directory / "gradient.png"
        utils.save_image(self.image, path)
        loaded = utils.load_image(path)
        np.testing.assert_allclose(loaded, self.image, atol=1 / 255)

    def test_loaded_image_is_float_in_unit_range(self) -> None:
        path = self.directory / "gradient.png"
        utils.save_image(self.image, path)
        loaded = utils.load_image(path)
        self.assertEqual(loaded.dtype, np.float32)
        self.assertGreaterEqual(float(loaded.min()), 0.0)
        self.assertLessEqual(float(loaded.max()), 1.0)

    def test_save_creates_missing_parent_directories(self) -> None:
        path = self.directory / "nested" / "deeper" / "image.png"
        utils.save_image(self.image, path)
        self.assertTrue(path.exists())


class TestRawRouting(unittest.TestCase):
    def test_raw_extensions_are_detected(self) -> None:
        self.assertTrue(utils.is_raw_file(Path("shot.CR3")))
        self.assertTrue(utils.is_raw_file(Path("shot.craw")))
        self.assertFalse(utils.is_raw_file(Path("shot.png")))

    def test_load_image_routes_raw_files_to_raw_decoder(self) -> None:
        fake = np.zeros((4, 4, 3), dtype=np.float32)
        with mock.patch("utils._load_raw_image", return_value=fake) as raw_loader:
            result = utils.load_image(Path("shot.cr3"))
        raw_loader.assert_called_once_with(Path("shot.cr3"))
        np.testing.assert_array_equal(result, fake)

    def test_load_image_routes_standard_files_to_pillow(self) -> None:
        fake = np.zeros((4, 4, 3), dtype=np.float32)
        with mock.patch(
            "utils._load_standard_image", return_value=fake
        ) as standard_loader:
            utils.load_image(Path("shot.jpg"))
        standard_loader.assert_called_once_with(Path("shot.jpg"))


if __name__ == "__main__":
    unittest.main()
