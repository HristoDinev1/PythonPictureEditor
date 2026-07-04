import threading
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import ImageTk

import image_processor
import utils
import timelapse
from main import Settings, apply_settings
from film_simulation import FilmPreset, load_presets

PREVIEW_MAX_SIZE: int = 640
PREVIEW_DEBOUNCE_MS: int = 60
NO_PRESET_LABEL: str = "None"
SLIDER_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Light": [
        ("Exposure", "exposure"),
        ("Contrast", "contrast"),
        ("Highlights", "highlights"),
        ("Shadows", "shadows"),
        ("Whites", "whites"),
        ("Blacks", "blacks"),
    ],
    "Color": [
        ("Temperature", "temperature"),
        ("Tint", "tint"),
        ("Vibrance", "vibrance"),
        ("Saturation", "saturation"),
    ],
    "Effects": [
        ("Texture", "texture"),
        ("Clarity", "clarity"),
        ("Dehaze", "dehaze"),
        ("Vignette", "vignette"),
    ],
    "Detail": [
        ("Noise Reduction (Lum)", "noise_reduction_luminance"),
        ("Noise Reduction (Color)", "noise_reduction_color"),
    ],
}


def downscale_for_preview(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_size:
        return image
    step = int(np.ceil(largest / max_size))
    return np.ascontiguousarray(image[::step, ::step])


class PhotoEditorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Batch Photo Editor + Timelapse")
        self.photo_paths: list[Path] = []
        self.preview_base: np.ndarray | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.presets: dict[str, FilmPreset] = load_presets()
        self.slider_variables: dict[str, tk.DoubleVar] = {}
        self.preset_variable = tk.StringVar(value=NO_PRESET_LABEL)
        self.status_variable = tk.StringVar(value="Load a folder to begin")
        self.render_generation = 0
        self.pending_render: str | None = None
        self._build_layout()

    def _build_layout(self) -> None:
        left = ttk.Frame(self.root, padding=4)
        left.grid(row=0, column=0, sticky="nsew")
        middle = ttk.Frame(self.root, padding=4)
        middle.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(self.root, padding=4)
        right.grid(row=0, column=2, sticky="nsew")
        bottom = ttk.Frame(self.root, padding=4)
        bottom.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Button(left, text="Load Folder…", command=self.load_folder).pack(fill="x")
        self.photo_list = tk.Listbox(left, width=32)
        self.photo_list.pack(fill="both", expand=True)
        self.photo_list.bind("<<ListboxSelect>>", self._on_photo_selected)

        self.preview_label = ttk.Label(middle, anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        for group_name, entries in SLIDER_GROUPS.items():
            frame = ttk.LabelFrame(right, text=group_name, padding=2)
            frame.pack(fill="x", pady=2)
            for label, field_name in entries:
                self._add_slider(frame, label, field_name)
        preset_frame = ttk.LabelFrame(right, text="Film Simulation", padding=2)
        preset_frame.pack(fill="x", pady=2)
        preset_box = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_variable,
            values=[NO_PRESET_LABEL, *self.presets],
            state="readonly",
        )
        preset_box.pack(fill="x")
        preset_box.bind("<<ComboboxSelected>>", lambda event: self.schedule_preview())

        ttk.Button(bottom, text="Apply to All Photos", command=self.apply_to_all).pack(
            side="left", padx=4
        )
        ttk.Button(bottom, text="Export Timelapse", command=self.export_timelapse).pack(
            side="left", padx=4
        )
        ttk.Label(bottom, textvariable=self.status_variable).pack(side="left", padx=8)

    def _add_slider(self, parent: ttk.LabelFrame, label: str, field_name: str) -> None:
        variable = tk.DoubleVar(value=0.0)
        self.slider_variables[field_name] = variable
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=label, width=22).pack(side="left")
        scale = ttk.Scale(
            row,
            from_=-100.0,
            to=100.0,
            variable=variable,
            command=lambda value: self.schedule_preview(),
        )
        scale.pack(side="left", fill="x", expand=True)

    def current_settings(self) -> Settings:
        values = {
            field.name: self.slider_variables[field.name].get()
            for field in fields(Settings)
            if field.name in self.slider_variables
        }
        return Settings(**values, film_simulation=self.preset_variable.get())

    def current_preset(self) -> FilmPreset | None:
        return self.presets.get(self.preset_variable.get())

    def load_folder(self) -> None:
        chosen = filedialog.askdirectory(title="Choose a photo folder")
        if not chosen:
            return
        self.photo_paths = utils.list_photo_files(Path(chosen))
        self.photo_list.delete(0, tk.END)
        for path in self.photo_paths:
            self.photo_list.insert(tk.END, path.name)
        if not self.photo_paths:
            self.status_variable.set("Folder contains no supported photos")
            return
        self.status_variable.set(f"Loaded {len(self.photo_paths)} photos")
        self.photo_list.selection_set(0)
        self._on_photo_selected(None)

    def _on_photo_selected(self, event: tk.Event | None) -> None:
        selection = self.photo_list.curselection()
        if not selection:
            return
        path = self.photo_paths[selection[0]]
        self.status_variable.set(f"Loading {path.name}…")

        def load() -> None:
            image = utils.load_image(path)
            preview = downscale_for_preview(image, PREVIEW_MAX_SIZE)
            self.root.after(0, lambda: self._set_preview_base(preview, path.name))

        threading.Thread(target=load, daemon=True).start()

    def _set_preview_base(self, image: np.ndarray, name: str) -> None:
        self.preview_base = image
        self.status_variable.set(f"Previewing {name}")
        self.schedule_preview()

    def schedule_preview(self) -> None:
        if self.pending_render is not None:
            self.root.after_cancel(self.pending_render)
        self.pending_render = self.root.after(
            PREVIEW_DEBOUNCE_MS, self._start_preview_render
        )

    def _start_preview_render(self) -> None:
        self.pending_render = None
        if self.preview_base is None:
            return
        self.render_generation += 1
        generation = self.render_generation
        base = self.preview_base
        settings = self.current_settings()
        preset = self.current_preset()

        def render() -> None:
            edited = apply_settings(base, settings, preset)
            self.root.after(0, lambda: self._show_preview(edited, generation))

        threading.Thread(target=render, daemon=True).start()

    def _show_preview(self, image: np.ndarray, generation: int) -> None:
        if generation != self.render_generation:
            return
        self.preview_photo = ImageTk.PhotoImage(utils.to_pil_image(image))
        self.preview_label.configure(image=self.preview_photo)

    def apply_to_all(self) -> None:
        if not self.photo_paths:
            messagebox.showinfo("Nothing loaded", "Load a folder of photos first")
            return
        chosen = filedialog.askdirectory(title="Choose an output folder")
        if not chosen:
            return
        output_directory = Path(chosen)
        settings = self.current_settings()
        preset = self.current_preset()
        self.status_variable.set("Processing batch…")

        def run() -> None:
            frames = image_processor.process_files(
                self.photo_paths, settings, output_directory, preset
            )
            self.root.after(
                0,
                lambda: self.status_variable.set(
                    f"Saved {len(frames)} frames to {output_directory}"
                ),
            )

        threading.Thread(target=run, daemon=True).start()

    def export_timelapse(self) -> None:
        chosen = filedialog.askdirectory(title="Folder with rendered frames")
        if not chosen:
            return
        frames_directory = Path(chosen)
        frame_paths = sorted(frames_directory.glob("frame_*.png"))
        if not frame_paths:
            messagebox.showinfo("No frames", "Apply settings to all photos first")
            return
        self.status_variable.set("Exporting timelapse…")

        def run() -> None:
            output = timelapse.export_timelapse(frame_paths, frames_directory)
            self.root.after(
                0, lambda: self.status_variable.set(f"Timelapse saved: {output}")
            )

        threading.Thread(target=run, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    PhotoEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
