import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import utils
import timelapse


class TestMp4Export(unittest.TestCase):
    def setUp(self) -> None:
        self.frames_directory = Path("/frames")
        self.output_path = Path("/out/timelapse.mp4")

    def test_ffmpeg_is_invoked_with_frame_pattern_and_fps(self) -> None:
        with mock.patch("timelapse.subprocess.run") as mock_run:
            result = timelapse.export_mp4(
                self.frames_directory, self.output_path, fps=30
            )
        self.assertEqual(result, self.output_path)
        mock_run.assert_called_once()
        command = mock_run.call_args.args[0]
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("30", command)
        self.assertIn(str(self.frames_directory / "frame_%04d.png"), command)
        self.assertIn(str(self.output_path), command)

    def test_ffmpeg_is_run_with_error_checking(self) -> None:
        with mock.patch("timelapse.subprocess.run") as mock_run:
            timelapse.export_mp4(self.frames_directory, self.output_path)
        self.assertTrue(mock_run.call_args.kwargs["check"])




class TestEstimateDuration(unittest.TestCase):
    def test_standard_mode_is_one_frame_per_photo(self) -> None:
        self.assertAlmostEqual(
            timelapse.estimate_duration_seconds(48, timelapse.MODE_STANDARD, fps=24),
            2.0,
        )

    def test_hyperlapse_mode_is_one_frame_per_photo(self) -> None:
        self.assertAlmostEqual(
            timelapse.estimate_duration_seconds(48, timelapse.MODE_HYPERLAPSE, fps=24),
            2.0,
        )

    def test_long_exposure_mode_divides_by_the_blend_window(self) -> None:
        duration = timelapse.estimate_duration_seconds(
            25, timelapse.MODE_LONG_EXPOSURE, fps=24, long_exposure_window=10
        )
        self.assertAlmostEqual(duration, 3 / 24)

    def test_zero_photos_is_zero_duration(self) -> None:
        self.assertEqual(
            timelapse.estimate_duration_seconds(0, timelapse.MODE_STANDARD), 0.0
        )


class TestBlendLongExposure(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.frame_paths = []
        for index in range(5):
            image = np.zeros((6, 6, 3), dtype=np.float32)
            image[index % 6, index % 6] = 1.0
            path = self.directory / f"frame_{index:04d}.png"
            utils.save_image(image, path)
            self.frame_paths.append(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_lighten_blend_keeps_the_brightest_pixel_from_every_frame(self) -> None:
        output_dir = self.directory / "blended"
        blended_paths = timelapse.blend_long_exposure(
            self.frame_paths, output_dir, window=5
        )
        self.assertEqual(len(blended_paths), 1)
        result = utils.load_image(blended_paths[0])
        for index in range(5):
            self.assertGreater(result[index % 6, index % 6].max(), 0.9)

    def test_window_size_controls_output_frame_count(self) -> None:
        output_dir = self.directory / "blended"
        blended_paths = timelapse.blend_long_exposure(
            self.frame_paths, output_dir, window=2
        )
        self.assertEqual(len(blended_paths), 3)

    def test_empty_frame_list_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            timelapse.blend_long_exposure([], self.directory / "blended")


class TestStabilizeHyperlapse(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _save_frames(self, images: list[np.ndarray]) -> list[Path]:
        paths = []
        for index, image in enumerate(images):
            path = self.directory / f"frame_{index:04d}.png"
            utils.save_image(image, path)
            paths.append(path)
        return paths

    def test_panning_sequence_is_aligned_to_a_common_view(self) -> None:
        generator = np.random.default_rng(0)
        height, width = 60, 90
        background = generator.random((height, width, 3), dtype=np.float32) * 0.1
        background[20:30, 35:50] = 0.9
        shifts = [(0, 0), (2, 4), (4, 8), (6, 12)]
        frames = [
            np.roll(background, shift=shift, axis=(0, 1)) for shift in shifts
        ]

        output_dir = self.directory / "stabilized"
        frame_paths = self._save_frames(frames)
        aligned_paths = timelapse.stabilize_hyperlapse(frame_paths, output_dir)

        self.assertEqual(len(aligned_paths), len(frames))
        aligned = [utils.load_image(path) for path in aligned_paths]
        base = aligned[0]
        for frame in aligned[1:]:
            self.assertLess(float(np.abs(frame - base).mean()), 1e-3)

    def test_single_frame_passes_through_unchanged(self) -> None:
        image = np.full((10, 10, 3), 0.4, dtype=np.float32)
        frame_paths = self._save_frames([image])
        output_dir = self.directory / "stabilized"
        aligned_paths = timelapse.stabilize_hyperlapse(frame_paths, output_dir)
        self.assertEqual(len(aligned_paths), 1)
        np.testing.assert_allclose(
            utils.load_image(aligned_paths[0]), image, atol=1e-2
        )

    def test_empty_frame_list_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            timelapse.stabilize_hyperlapse([], self.directory / "stabilized")


class TestPrepareFrames(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.frame_paths = []
        for index in range(3):
            image = np.full((6, 6, 3), (index + 1) / 4, dtype=np.float32)
            path = self.directory / f"frame_{index:04d}.png"
            utils.save_image(image, path)
            self.frame_paths.append(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_standard_mode_returns_paths_unchanged(self) -> None:
        result = timelapse.prepare_frames(
            self.frame_paths, timelapse.MODE_STANDARD, self.directory / "out"
        )
        self.assertEqual(result, self.frame_paths)

    def test_long_exposure_mode_blends(self) -> None:
        result = timelapse.prepare_frames(
            self.frame_paths,
            timelapse.MODE_LONG_EXPOSURE,
            self.directory / "out",
            long_exposure_window=3,
        )
        self.assertEqual(len(result), 1)

    def test_hyperlapse_mode_stabilizes(self) -> None:
        result = timelapse.prepare_frames(
            self.frame_paths, timelapse.MODE_HYPERLAPSE, self.directory / "out"
        )
        self.assertEqual(len(result), 3)


class TestExportTimelapse(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        image = np.full((8, 8, 3), 0.5, dtype=np.float32)
        self.frame_path = self.directory / "frame_0000.png"
        utils.save_image(image, self.frame_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exports_mp4_when_ffmpeg_is_available(self) -> None:
        with (
            mock.patch("timelapse.ffmpeg_available", return_value=True),
            mock.patch("timelapse.subprocess.run") as mock_run,
        ):
            output = timelapse.export_timelapse([self.frame_path], self.directory)
        self.assertEqual(output.suffix, ".mp4")
        mock_run.assert_called_once()

    def test_raises_runtime_error_without_ffmpeg(self) -> None:
        with mock.patch("timelapse.ffmpeg_available", return_value=False):
            with self.assertRaises(RuntimeError):
                timelapse.export_timelapse([self.frame_path], self.directory)

    def test_no_frames_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            timelapse.export_timelapse([], self.directory)


if __name__ == "__main__":
    unittest.main()
