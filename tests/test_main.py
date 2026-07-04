import unittest
from collections.abc import Callable
from unittest import mock

import numpy as np

import main
from film_simulation import FilmPreset

ADJUSTMENTS: list[Callable[[np.ndarray, float], np.ndarray]] = [
    main.adjust_exposure,
    main.adjust_contrast,
    main.adjust_highlights,
    main.adjust_shadows,
    main.adjust_whites,
    main.adjust_blacks,
    main.adjust_temperature,
    main.adjust_tint,
    main.adjust_vibrance,
    main.adjust_saturation,
    main.adjust_texture,
    main.adjust_clarity,
    main.adjust_dehaze,
    main.adjust_vignette,
    main.reduce_luminance_noise,
    main.reduce_color_noise,
]


def make_test_image(height: int = 48, width: int = 64) -> np.ndarray:
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    red = rows * np.ones_like(cols)
    green = np.ones_like(rows) * cols
    blue = (rows + cols) / 2.0
    return np.stack([red, green, blue], axis=-1)


class TestAdjustmentBasics(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_test_image()

    def test_zero_amount_is_no_op_for_every_adjustment(self) -> None:
        for adjustment in ADJUSTMENTS:
            result = adjustment(self.image, 0.0)
            np.testing.assert_array_equal(result, self.image)

    def test_extreme_amounts_stay_within_unit_range(self) -> None:
        for adjustment in ADJUSTMENTS:
            for amount in (-1.0, 1.0):
                result = adjustment(self.image, amount)
                self.assertGreaterEqual(float(result.min()), 0.0)
                self.assertLessEqual(float(result.max()), 1.0)

    def test_zero_amount_does_not_mutate_input(self) -> None:
        original = self.image.copy()
        for adjustment in ADJUSTMENTS:
            adjustment(self.image, 0.0)
        np.testing.assert_array_equal(self.image, original)


class TestTonalAdjustments(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_test_image()
        self.mean = float(self.image.mean())

    def test_positive_exposure_brightens(self) -> None:
        brightened = main.adjust_exposure(self.image, 0.5)
        self.assertGreater(float(brightened.mean()), self.mean)

    def test_negative_exposure_darkens(self) -> None:
        darkened = main.adjust_exposure(self.image, -0.5)
        self.assertLess(float(darkened.mean()), self.mean)

    def test_positive_contrast_spreads_values(self) -> None:
        result = main.adjust_contrast(self.image, 0.8)
        self.assertGreater(float(result.std()), float(self.image.std()))

    def test_highlights_leave_deep_shadows_untouched(self) -> None:
        dark = np.full((8, 8, 3), 0.05, dtype=np.float32)
        result = main.adjust_highlights(dark, -1.0)
        np.testing.assert_allclose(result, dark, atol=1e-6)

    def test_shadows_leave_bright_areas_untouched(self) -> None:
        bright = np.full((8, 8, 3), 0.95, dtype=np.float32)
        result = main.adjust_shadows(bright, 1.0)
        np.testing.assert_allclose(result, bright, atol=1e-6)


class TestColorAdjustments(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_test_image()

    def test_warm_temperature_raises_red_over_blue(self) -> None:
        result = main.adjust_temperature(self.image, 0.5)
        self.assertGreater(
            float(result[..., 0].mean()), float(self.image[..., 0].mean())
        )
        self.assertLess(float(result[..., 2].mean()), float(self.image[..., 2].mean()))

    def test_full_desaturation_makes_channels_equal(self) -> None:
        result = main.adjust_saturation(self.image, -1.0)
        np.testing.assert_allclose(result[..., 0], result[..., 1], atol=1e-5)
        np.testing.assert_allclose(result[..., 1], result[..., 2], atol=1e-5)

    def test_vibrance_boosts_muted_pixels_more_than_saturated(self) -> None:
        muted = np.full((4, 4, 3), 0.5, dtype=np.float32)
        muted[..., 0] = 0.55
        saturated = np.zeros((4, 4, 3), dtype=np.float32)
        saturated[..., 0] = 1.0
        muted_gain = float(
            (main.adjust_vibrance(muted, 1.0) - muted)[..., 0].mean()
        )
        saturated_gain = float(
            (main.adjust_vibrance(saturated, 1.0) - saturated)[..., 0].mean()
        )
        self.assertGreater(muted_gain, saturated_gain)


class TestDehaze(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(7)
        hazy = 0.4 + 0.2 * generator.random((64, 64), dtype=np.float32)
        self.image = np.stack([hazy, hazy, hazy], axis=-1)

    def _histogram_spread(self, gray: np.ndarray) -> float:
        histogram, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
        return float(np.std(histogram / histogram.sum()))

    def test_equalized_histogram_is_flatter_than_input(self) -> None:
        before = self._histogram_spread(main.luminance(self.image))
        equalized = main.tiled_equalize(main.luminance(self.image))
        self.assertLess(self._histogram_spread(equalized), before)

    def test_dehaze_output_stays_in_unit_range(self) -> None:
        result = main.adjust_dehaze(self.image, 1.0)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_positive_dehaze_increases_luminance_spread(self) -> None:
        before = float(main.luminance(self.image).std())
        after = float(main.luminance(main.adjust_dehaze(self.image, 1.0)).std())
        self.assertGreater(after, before)


class TestFilmSimulation(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_test_image()
        self.identity = FilmPreset(
            name="Identity",
            tone_curve=[[0.0, 0.0], [1.0, 1.0]],
            color_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            grain_amount=0.0,
            grain_seed=1,
        )

    def test_identity_preset_changes_nothing(self) -> None:
        result = main.apply_film_preset(self.image, self.identity)
        np.testing.assert_allclose(result, self.image, atol=1e-6)

    def test_monochrome_matrix_equalizes_channels(self) -> None:
        row = [0.2126, 0.7152, 0.0722]
        result = main.apply_color_matrix(self.image, [row, row, row])
        np.testing.assert_allclose(result[..., 0], result[..., 1], atol=1e-6)
        np.testing.assert_allclose(result[..., 1], result[..., 2], atol=1e-6)

    def test_grain_is_deterministic_for_a_fixed_seed(self) -> None:
        first = main.apply_grain(self.image, 0.05, seed=42)
        second = main.apply_grain(self.image, 0.05, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_grain_uses_the_preset_seed(self) -> None:
        with mock.patch("main.np.random.default_rng") as mock_rng:
            mock_rng.return_value.normal.return_value = np.zeros(
                self.image.shape[:2], dtype=np.float32
            )
            main.apply_grain(self.image, 0.05, seed=1234)
        mock_rng.assert_called_once_with(1234)

    def test_zero_grain_skips_the_rng_entirely(self) -> None:
        with mock.patch("main.np.random.default_rng") as mock_rng:
            main.apply_grain(self.image, 0.0, seed=1234)
        mock_rng.assert_not_called()


class TestApplySettings(unittest.TestCase):
    def setUp(self) -> None:
        self.image = make_test_image()

    def test_default_settings_are_a_no_op(self) -> None:
        result = main.apply_settings(self.image, main.Settings())
        np.testing.assert_allclose(result, self.image, atol=1e-6)

    def test_extreme_settings_stay_in_unit_range(self) -> None:
        maxed = main.Settings(
            exposure=100.0,
            contrast=100.0,
            highlights=-100.0,
            shadows=100.0,
            whites=100.0,
            blacks=-100.0,
            temperature=100.0,
            tint=-100.0,
            vibrance=100.0,
            saturation=100.0,
            texture=100.0,
            clarity=100.0,
            dehaze=100.0,
            vignette=100.0,
            noise_reduction_luminance=100.0,
            noise_reduction_color=100.0,
        )
        result = main.apply_settings(self.image, maxed)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
