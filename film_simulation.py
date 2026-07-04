import json
from dataclasses import dataclass
from pathlib import Path

PRESETS_DIRECTORY: Path = Path(__file__).parent / "presets"


@dataclass(frozen=True)
class FilmPreset:
    name: str
    tone_curve: list[list[float]]
    color_matrix: list[list[float]]
    grain_amount: float
    grain_seed: int


def _validate_tone_curve(curve: object) -> list[list[float]]:
    if not isinstance(curve, list) or len(curve) < 2:
        raise ValueError("tone_curve must be a list of at least two [x, y] points")
    points: list[list[float]] = []
    for point in curve:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("each tone_curve point must be an [x, y] pair")
        x, y = float(point[0]), float(point[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("tone_curve points must lie within [0, 1]")
        points.append([x, y])
    xs = [point[0] for point in points]
    if xs != sorted(xs):
        raise ValueError("tone_curve x values must be non-decreasing")
    return points


def _validate_color_matrix(matrix: object) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError("color_matrix must have exactly three rows")
    rows: list[list[float]] = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("each color_matrix row must have exactly three values")
        rows.append([float(value) for value in row])
    return rows


def parse_preset(data: object) -> FilmPreset:
    if not isinstance(data, dict):
        raise ValueError("preset must be a JSON object")
    required = {"name", "tone_curve", "color_matrix", "grain_amount", "grain_seed"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"preset is missing keys: {sorted(missing)}")
    name = data["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("preset name must be a non-empty string")
    grain_amount = float(data["grain_amount"])
    if not 0.0 <= grain_amount <= 1.0:
        raise ValueError("grain_amount must be within [0, 1]")
    return FilmPreset(
        name=name,
        tone_curve=_validate_tone_curve(data["tone_curve"]),
        color_matrix=_validate_color_matrix(data["color_matrix"]),
        grain_amount=grain_amount,
        grain_seed=int(data["grain_seed"]),
    )


def load_preset(path: Path) -> FilmPreset:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed preset JSON in {path.name}: {error}") from error
    return parse_preset(data)


def load_presets(directory: Path = PRESETS_DIRECTORY) -> dict[str, FilmPreset]:
    presets: dict[str, FilmPreset] = {}
    for path in sorted(directory.glob("*.json")):
        preset = load_preset(path)
        presets[preset.name] = preset
    return presets
