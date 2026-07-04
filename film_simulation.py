from dataclasses import dataclass


@dataclass(frozen=True)
class FilmPreset:
    name: str
    tone_curve: list[list[float]]
    color_matrix: list[list[float]]
    grain_amount: float
    grain_seed: int


KODACHROME_64 = FilmPreset(
    name="Kodachrome 64",
    tone_curve=[[0.0, 0.02], [0.25, 0.22], [0.5, 0.52], [0.75, 0.8], [1.0, 0.98]],
    color_matrix=[[1.08, -0.04, -0.04], [-0.02, 1.04, -0.02], [-0.04, -0.06, 1.1]],
    grain_amount=0.02,
    grain_seed=64,
)

PORTRA_400 = FilmPreset(
    name="Portra 400",
    tone_curve=[[0.0, 0.04], [0.3, 0.32], [0.6, 0.64], [1.0, 0.96]],
    color_matrix=[[1.04, 0.02, -0.02], [0.01, 1.0, 0.01], [-0.02, 0.02, 1.02]],
    grain_amount=0.015,
    grain_seed=400,
)

TRI_X_400 = FilmPreset(
    name="Tri-X 400",
    tone_curve=[[0.0, 0.0], [0.2, 0.14], [0.5, 0.5], [0.8, 0.86], [1.0, 1.0]],
    color_matrix=[
        [0.2126, 0.7152, 0.0722],
        [0.2126, 0.7152, 0.0722],
        [0.2126, 0.7152, 0.0722],
    ],
    grain_amount=0.045,
    grain_seed=1954,
)

VELVIA_50 = FilmPreset(
    name="Velvia 50",
    tone_curve=[[0.0, 0.0], [0.25, 0.18], [0.5, 0.5], [0.75, 0.84], [1.0, 1.0]],
    color_matrix=[[1.16, -0.08, -0.08], [-0.06, 1.14, -0.08], [-0.06, -0.08, 1.16]],
    grain_amount=0.01,
    grain_seed=50,
)

FILM_PRESETS: dict[str, FilmPreset] = {
    preset.name: preset
    for preset in (KODACHROME_64, PORTRA_400, TRI_X_400, VELVIA_50)
}
