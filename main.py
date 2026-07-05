from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.ndimage import minimum_filter as _scipy_minimum_filter
except ImportError:
    _scipy_minimum_filter = None

from film_simulation import FilmPreset

LUMA_WEIGHTS: np.ndarray = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class Settings:
    exposure: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    milky_way_glow: float = 0.0
    vignette: float = 0.0
    sharpness: float = 0.0
    noise_reduction_luminance: float = 0.0
    noise_reduction_color: float = 0.0
    rotation: int = 0
    film_simulation: str = field(default="None")


def clip_unit(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0)


def apply_rotation(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees % 360 == 0:
        return image
    steps = (-degrees // 90) % 4
    return np.ascontiguousarray(np.rot90(image, k=steps))


def luminance(image: np.ndarray) -> np.ndarray:
    return image @ LUMA_WEIGHTS


def smoothstep(values: np.ndarray, edge_low: float, edge_high: float) -> np.ndarray:
    ramp = np.clip((values - edge_low) / (edge_high - edge_low), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _box_blur_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    window = 2 * radius + 1
    pad_width = [
        (radius, radius) if index == axis else (0, 0) for index in range(values.ndim)
    ]
    padded = np.pad(values, pad_width, mode="edge")
    zero_shape = list(padded.shape)
    zero_shape[axis] = 1
    running = np.concatenate(
        [
            np.zeros(zero_shape, dtype=np.float32),
            np.cumsum(padded, axis=axis, dtype=np.float32),
        ],
        axis=axis,
    )
    upper = np.take(running, range(window, running.shape[axis]), axis=axis)
    lower = np.take(running, range(0, running.shape[axis] - window), axis=axis)
    return ((upper - lower) / window).astype(np.float32)


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image
    blurred = _box_blur_axis(image, radius, axis=0)
    return _box_blur_axis(blurred, radius, axis=1)


def _min_filter_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    pad_width = [
        (radius, radius) if index == axis else (0, 0) for index in range(values.ndim)
    ]
    padded = np.pad(values, pad_width, mode="edge")
    result: np.ndarray | None = None
    for shift in range(2 * radius + 1):
        window = np.take(
            padded, range(shift, shift + values.shape[axis]), axis=axis
        )
        result = window if result is None else np.minimum(result, window)
    assert result is not None
    return result


def min_filter(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values
    if _scipy_minimum_filter is not None:
        return _scipy_minimum_filter(values, size=2 * radius + 1, mode="nearest")
    return _min_filter_axis(_min_filter_axis(values, radius, 0), radius, 1)


def adjust_exposure(image: np.ndarray, stops: float) -> np.ndarray:
    if stops == 0.0:
        return image
    return clip_unit(image * float(2.0**stops))


def adjust_brightness(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    gamma = 2.0 ** (-amount)
    return clip_unit(np.power(np.clip(image, 0.0, 1.0), gamma))


def adjust_contrast(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return clip_unit((image - 0.5) * float(2.0**amount) + 0.5)


def _tonal_shift(
    image: np.ndarray,
    amount: float,
    edge_low: float,
    edge_high: float,
    invert: bool = False,
    min_width: float = 0.0,
) -> np.ndarray:
    weight = smoothstep(luminance(image), edge_low, edge_high)
    if invert:
        weight = 1.0 - weight
    # Capping max_delta at (window / 3) is what keeps this monotonic (see below),
    # but a window derived from this image's own percentiles can be much narrower
    # than the window a fixed threshold would use -- min_width keeps the *delta*
    # from collapsing to near-nothing just because the matching pixels happen to
    # be clustered close together, without changing which pixels are targeted.
    width = max(edge_high - edge_low, min_width)
    max_delta = width / 3.0
    return clip_unit(image + max_delta * amount * weight[..., None])


def adjust_highlights(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _tonal_shift(image, amount, 0.15, 1.0)


def adjust_shadows(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _tonal_shift(image, amount, 0.0, 0.5, invert=True)


def adjust_whites(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    # Fixed at 0.75..1.0, this never touched anything on a photo whose brightest
    # content (e.g. a night sky's stars) never reaches that range -- the slider
    # would look like it did nothing. Anchoring the window to this image's own
    # brightest percentiles instead means it always has real content to act on;
    # min_width keeps the effect visible even when that content is clustered in
    # a narrow band, which is exactly the case on a low-dynamic-range photo.
    gray = luminance(image)
    edge_high = float(np.percentile(gray, 99.5))
    edge_low = float(np.percentile(gray, 85.0))
    if edge_high <= edge_low:
        edge_high = edge_low + 1e-6
    return _tonal_shift(image, amount, edge_low, edge_high, min_width=0.15)


def adjust_blacks(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _tonal_shift(image, amount, 0.0, 0.25, invert=True)


def adjust_temperature(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    scale = np.array(
        [1.0 + 0.25 * amount, 1.0, 1.0 - 0.25 * amount], dtype=np.float32
    )
    return clip_unit(image * scale)


def adjust_tint(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    scale = np.array(
        [1.0 - 0.1 * amount, 1.0 + 0.2 * amount, 1.0 - 0.1 * amount],
        dtype=np.float32,
    )
    return clip_unit(image * scale)


def adjust_saturation(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    gray = luminance(image)[..., None]
    return clip_unit(gray + (image - gray) * (1.0 + amount))


def adjust_vibrance(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    saturation_level = image.max(axis=-1) - image.min(axis=-1)
    boost = 1.0 + amount * (1.0 - saturation_level)[..., None]
    gray = luminance(image)[..., None]
    return clip_unit(gray + (image - gray) * boost)


def _local_contrast(image: np.ndarray, amount: float, radius: int) -> np.ndarray:
    detail = image - box_blur(image, radius)
    return clip_unit(image + amount * detail)


def adjust_sharpness(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _local_contrast(image, 2.0 * amount, radius=1)


def adjust_texture(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    return _local_contrast(image, 1.5 * amount, radius=2)


def adjust_clarity(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    radius = max(4, min(image.shape[0], image.shape[1]) // 50)
    return _local_contrast(image, amount, radius)


HAZE_VEIL: np.ndarray = np.array([0.8, 0.82, 0.85], dtype=np.float32)


def _dehaze_radius(image: np.ndarray) -> int:
    return max(1, min(image.shape[0], image.shape[1]) // 40)


def _remove_haze(image: np.ndarray, amount: float) -> np.ndarray:
    radius = _dehaze_radius(image)
    dark_channel = min_filter(image.min(axis=-1), radius)
    flat = dark_channel.ravel()
    sample = max(1, flat.size // 1000)
    haziest = np.argpartition(flat, -sample)[-sample:]
    airlight = image.reshape(-1, 3)[haziest].mean(axis=0)
    airlight = np.clip(airlight, 0.15, 1.0)
    normalized = min_filter((image / airlight).min(axis=-1), radius)
    transmission = np.maximum(1.0 - 0.95 * amount * normalized, 0.1)
    recovered = (image - airlight) / transmission[..., None] + airlight
    return clip_unit(recovered)


def _add_haze(image: np.ndarray, amount: float) -> np.ndarray:
    transmission = 1.0 - 0.6 * amount
    return clip_unit(image * transmission + HAZE_VEIL * (1.0 - transmission))


def adjust_dehaze(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    if amount > 0.0:
        return _remove_haze(image, amount)
    return _add_haze(image, -amount)


MILKY_WAY_GLOW_RADIUS_DIVISOR: int = 24
MILKY_WAY_GLOW_STRENGTH: float = 2.0


def adjust_milky_way_glow(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    gray = luminance(image)
    radius = max(4, min(image.shape[0], image.shape[1]) // MILKY_WAY_GLOW_RADIUS_DIVISOR)
    local_average = box_blur(gray, radius)
    baseline = np.percentile(local_average, 20)
    structure = local_average - baseline
    high = np.percentile(structure, 99)
    weight = smoothstep(structure, 0.0, max(float(high), 1e-6))
    boost = 1.0 + weight * amount * MILKY_WAY_GLOW_STRENGTH
    return clip_unit(image * boost[..., None])


def adjust_vignette(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    height, width = image.shape[:2]
    rows = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    distance = np.sqrt(rows * rows + cols * cols) / np.sqrt(2.0)
    falloff = smoothstep(distance, 0.3, 1.0)
    return clip_unit(image * (1.0 - amount * falloff)[..., None])


def reduce_luminance_noise(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    gray = luminance(image)
    smoothed = gray + amount * (box_blur(gray, radius=2) - gray)
    delta = smoothed - gray
    return clip_unit(image + delta[..., None])


def reduce_color_noise(image: np.ndarray, amount: float) -> np.ndarray:
    if amount == 0.0:
        return image
    radius = max(3, min(image.shape[0], image.shape[1]) // 400)
    gray = luminance(image)[..., None]
    chroma = image - gray
    smoothed_chroma = chroma + amount * (box_blur(chroma, radius) - chroma)
    return clip_unit(gray + smoothed_chroma)


def apply_tone_curve(image: np.ndarray, curve: list[list[float]]) -> np.ndarray:
    points = np.array(curve, dtype=np.float32)
    return np.interp(image, points[:, 0], points[:, 1]).astype(np.float32)


def apply_color_matrix(image: np.ndarray, matrix: list[list[float]]) -> np.ndarray:
    return clip_unit(image @ np.array(matrix, dtype=np.float32).T)


def apply_grain(image: np.ndarray, amount: float, seed: int) -> np.ndarray:
    if amount == 0.0:
        return image
    generator = np.random.default_rng(seed)
    noise = generator.normal(0.0, amount, size=image.shape[:2]).astype(np.float32)
    return clip_unit(image + noise[..., None])


def apply_film_preset(image: np.ndarray, preset: FilmPreset) -> np.ndarray:
    curved = apply_tone_curve(image, preset.tone_curve)
    graded = apply_color_matrix(curved, preset.color_matrix)
    return apply_grain(graded, preset.grain_amount, preset.grain_seed)


def apply_settings(
    image: np.ndarray,
    settings: Settings,
    preset: FilmPreset | None = None,
) -> np.ndarray:
    scaled = {
        name: getattr(settings, name) / 100.0
        for name in (
            "brightness",
            "contrast",
            "highlights",
            "shadows",
            "whites",
            "blacks",
            "temperature",
            "tint",
            "vibrance",
            "saturation",
            "texture",
            "clarity",
            "dehaze",
            "milky_way_glow",
            "vignette",
            "sharpness",
            "noise_reduction_luminance",
            "noise_reduction_color",
        )
    }
    result = apply_rotation(image.astype(np.float32), settings.rotation)
    result = adjust_exposure(result, settings.exposure)
    result = adjust_brightness(result, scaled["brightness"])
    result = adjust_temperature(result, scaled["temperature"])
    result = adjust_tint(result, scaled["tint"])
    result = adjust_contrast(result, scaled["contrast"])
    result = adjust_highlights(result, scaled["highlights"])
    result = adjust_shadows(result, scaled["shadows"])
    result = adjust_whites(result, scaled["whites"])
    result = adjust_blacks(result, scaled["blacks"])
    result = adjust_vibrance(result, scaled["vibrance"])
    result = adjust_saturation(result, scaled["saturation"])
    result = adjust_texture(result, scaled["texture"])
    result = adjust_clarity(result, scaled["clarity"])
    result = adjust_dehaze(result, scaled["dehaze"])
    result = adjust_milky_way_glow(result, scaled["milky_way_glow"])
    result = reduce_luminance_noise(result, scaled["noise_reduction_luminance"])
    result = reduce_color_noise(result, scaled["noise_reduction_color"])
    result = adjust_sharpness(result, scaled["sharpness"])
    result = adjust_vignette(result, scaled["vignette"])
    if preset is not None:
        result = apply_film_preset(result, preset)
    return result


if __name__ == "__main__":
    from gui import main as launch_gui

    launch_gui()
