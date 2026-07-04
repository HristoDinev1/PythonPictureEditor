import tempfile
import unittest
from pathlib import Path

import numpy as np

import image_processor
import utils
from main import Settings


def make_batch_images(count: int) -> list[np.ndarray]:
    images = []
    for index in range(count):
        level = (index + 1) / (count + 1)
        images.append(np.full((16, 16, 3), level, dtype=np.float32))
    return images


class TestProcessImages(unittest.TestCase):
    def setUp(self) -> None:
        self.images = make_batch_images(12)
        self.settings = Settings(exposure=40.0, contrast=25.0, vignette=30.0)

    def test_parallel_matches_serial_exactly(self) -> None:
        serial = image_processor.process_images_serial(self.images, self.settings)
        parallel = image_processor.process_images(self.images, self.settings, max_workers=4)
        self.assertEqual(len(serial), len(parallel))
        for expected, actual in zip(serial, parallel, strict=True):
            np.testing.assert_array_equal(expected, actual)

    def test_results_come_back_in_input_order(self) -> None:
        results = image_processor.process_images(self.images, Settings(), max_workers=4)
        for source, result in zip(self.images, results, strict=True):
            np.testing.assert_allclose(result, source, atol=1e-6)

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(image_processor.process_images([], self.settings), [])

    def test_single_worker_matches_many_workers(self) -> None:
        one = image_processor.process_images(self.images, self.settings, max_workers=1)
        many = image_processor.process_images(self.images, self.settings, max_workers=8)
        for expected, actual in zip(one, many, strict=True):
            np.testing.assert_array_equal(expected, actual)


class TestProcessFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.source_dir = Path(self.temp.name) / "source"
        self.output_dir = Path(self.temp.name) / "output"
        self.source_dir.mkdir()
        self.paths = []
        for index, image in enumerate(make_batch_images(5)):
            path = self.source_dir / f"photo_{index}.png"
            utils.save_image(image, path)
            self.paths.append(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_frames_are_numbered_in_input_order(self) -> None:
        frames = image_processor.process_files(
            self.paths, Settings(), self.output_dir, max_workers=4
        )
        expected = [self.output_dir / f"frame_{i:04d}.png" for i in range(5)]
        self.assertEqual(frames, expected)
        for frame in frames:
            self.assertTrue(frame.exists())

    def test_frame_content_matches_serial_pipeline(self) -> None:
        settings = Settings(saturation=-100.0)
        frames = image_processor.process_files(
            self.paths, settings, self.output_dir, max_workers=4
        )
        serial = image_processor.process_images_serial(
            [utils.load_image(path) for path in self.paths], settings
        )
        for frame, expected in zip(frames, serial, strict=True):
            saved_and_reloaded = utils.load_image(frame)
            np.testing.assert_allclose(saved_and_reloaded, expected, atol=1 / 255)

    def test_empty_path_list_returns_empty_list(self) -> None:
        self.assertEqual(image_processor.process_files([], Settings(), self.output_dir), [])


if __name__ == "__main__":
    unittest.main()
