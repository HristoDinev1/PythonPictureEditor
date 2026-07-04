import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

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


class TestGifExport(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.frame_paths = []
        for index in range(4):
            level = (index + 1) / 5.0
            image = np.full((8, 8, 3), level, dtype=np.float32)
            path = self.directory / f"frame_{index:04d}.png"
            utils.save_image(image, path)
            self.frame_paths.append(path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_gif_contains_every_frame(self) -> None:
        output = self.directory / "timelapse.gif"
        timelapse.export_gif(self.frame_paths, output, fps=10)
        with Image.open(output) as gif:
            self.assertEqual(gif.n_frames, 4)

    def test_empty_frame_list_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            timelapse.export_gif([], self.directory / "timelapse.gif")


class TestExportTimelapse(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        image = np.full((8, 8, 3), 0.5, dtype=np.float32)
        self.frame_path = self.directory / "frame_0000.png"
        utils.save_image(image, self.frame_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_prefers_mp4_when_ffmpeg_is_available(self) -> None:
        with (
            mock.patch("timelapse.ffmpeg_available", return_value=True),
            mock.patch("timelapse.subprocess.run") as mock_run,
        ):
            output = timelapse.export_timelapse([self.frame_path], self.directory)
        self.assertEqual(output.suffix, ".mp4")
        mock_run.assert_called_once()

    def test_falls_back_to_gif_without_ffmpeg(self) -> None:
        with mock.patch("timelapse.ffmpeg_available", return_value=False):
            output = timelapse.export_timelapse([self.frame_path], self.directory)
        self.assertEqual(output.suffix, ".gif")
        self.assertTrue(output.exists())

    def test_no_frames_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            timelapse.export_timelapse([], self.directory)


if __name__ == "__main__":
    unittest.main()
