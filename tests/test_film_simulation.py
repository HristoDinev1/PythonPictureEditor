import json
import tempfile
import unittest
from pathlib import Path

import film_simulation

VALID_PRESET: dict[str, object] = {
    "name": "Test Film",
    "tone_curve": [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]],
    "color_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "grain_amount": 0.02,
    "grain_seed": 99,
}


class TestPresetLoading(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, filename: str, content: str) -> Path:
        path = self.directory / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_valid_preset_loads_with_all_fields(self) -> None:
        path = self._write("film.json", json.dumps(VALID_PRESET))
        preset = film_simulation.load_preset(path)
        self.assertEqual(preset.name, "Test Film")
        self.assertEqual(preset.grain_seed, 99)
        self.assertEqual(len(preset.color_matrix), 3)

    def test_malformed_json_raises_value_error(self) -> None:
        path = self._write("broken.json", "{not valid json!!")
        with self.assertRaises(ValueError):
            film_simulation.load_preset(path)

    def test_missing_key_raises_value_error(self) -> None:
        incomplete = {
            key: value for key, value in VALID_PRESET.items() if key != "tone_curve"
        }
        path = self._write("incomplete.json", json.dumps(incomplete))
        with self.assertRaises(ValueError):
            film_simulation.load_preset(path)

    def test_non_square_color_matrix_raises_value_error(self) -> None:
        bad = dict(VALID_PRESET, color_matrix=[[1.0, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            film_simulation.parse_preset(bad)

    def test_unsorted_tone_curve_raises_value_error(self) -> None:
        bad = dict(VALID_PRESET, tone_curve=[[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
        with self.assertRaises(ValueError):
            film_simulation.parse_preset(bad)

    def test_tone_curve_point_outside_unit_range_raises_value_error(self) -> None:
        bad = dict(VALID_PRESET, tone_curve=[[0.0, 0.0], [1.5, 1.0]])
        with self.assertRaises(ValueError):
            film_simulation.parse_preset(bad)

    def test_load_presets_indexes_by_name(self) -> None:
        self._write("a.json", json.dumps(VALID_PRESET))
        other = dict(VALID_PRESET, name="Other Film")
        self._write("b.json", json.dumps(other))
        loaded = film_simulation.load_presets(self.directory)
        self.assertEqual(set(loaded), {"Test Film", "Other Film"})

    def test_empty_directory_yields_no_presets(self) -> None:
        self.assertEqual(film_simulation.load_presets(self.directory), {})


class TestBundledPresets(unittest.TestCase):
    def test_bundled_presets_are_all_valid(self) -> None:
        loaded = film_simulation.load_presets()
        self.assertGreaterEqual(len(loaded), 3)
        for preset in loaded.values():
            self.assertIsInstance(preset, film_simulation.FilmPreset)


if __name__ == "__main__":
    unittest.main()
