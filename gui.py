import ctypes
import sys
import threading
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import image_processor
import utils
import timelapse
from main import Settings, apply_settings
from film_simulation import FilmPreset, load_presets

PREVIEW_MAX_SIZE: int = 1100
PREVIEW_DEBOUNCE_MS: int = 60
NO_PRESET_LABEL: str = "None"

MIDNIGHT_BG: str = "#1c1e21"
MIDNIGHT_PANEL: str = "#25272b"
MIDNIGHT_BORDER: str = "#333539"
MIDNIGHT_TROUGH: str = "#2a2c30"
MIDNIGHT_ACCENT: str = "#3d76ac"
MIDNIGHT_ACCENT_ACTIVE: str = "#4c89c2"
MIDNIGHT_FG: str = "#c2c4c8"
MIDNIGHT_MUTED_FG: str = "#7c7e83"
MIDNIGHT_BUTTON_BG: str = "#3c3f45"
MIDNIGHT_BUTTON_HOVER: str = "#484b52"
MIDNIGHT_BUTTON_PRESSED: str = "#33363b"

DEFAULT_SLIDER_RANGE: tuple[float, float, int] = (-100.0, 100.0, 0)
SLIDER_RANGES: dict[str, tuple[float, float, int]] = {
    "exposure": (-5.0, 5.0, 1),
    "sharpness": (0.0, 150.0, 0),
    "noise_reduction_luminance": (0.0, 100.0, 0),
    "noise_reduction_color": (0.0, 100.0, 0),
}
SLIDER_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Light": [
        ("Exposure", "exposure"),
        ("Brightness", "brightness"),
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
        ("Milky Way Glow", "milky_way_glow"),
        ("Vignette", "vignette"),
    ],
    "Detail": [
        ("Sharpness", "sharpness"),
        ("Noise Reduction", "noise_reduction_luminance"),
        ("Noise Reduction (Color)", "noise_reduction_color"),
    ],
}
TIMELAPSE_FPS: int = 24
TIMELAPSE_MODE_LABELS: dict[str, str] = {
    "Standard Time-Lapse": timelapse.MODE_STANDARD,
    "Hyperlapse": timelapse.MODE_HYPERLAPSE,
    "Long-Exposure Time-Lapse": timelapse.MODE_LONG_EXPOSURE,
}


def downscale_for_preview(image: np.ndarray, max_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_size:
        return image
    step = int(np.ceil(largest / max_size))
    return np.ascontiguousarray(image[::step, ::step])


def fit_within_box(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    if max_width <= 1 or max_height <= 1:
        return image
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    if scale >= 1.0:
        return image
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


class PhotoEditorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Darkroom")
        self.root.geometry("1600x1000")
        self.root.minsize(1100, 700)
        self.photo_paths: list[Path] = []
        self.preview_base: np.ndarray | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_rendered_image: Image.Image | None = None
        self.pending_resize: str | None = None
        self.presets: dict[str, FilmPreset] = load_presets()
        self.slider_variables: dict[str, tk.DoubleVar] = {}
        self.slider_decimals: dict[str, int] = {}
        self.rotation: int = 0
        self.preset_variable = tk.StringVar(value=NO_PRESET_LABEL)
        self.status_variable = tk.StringVar(value='"Darkroom"')
        self.timelapse_mode_variable = tk.StringVar(
            value=next(iter(TIMELAPSE_MODE_LABELS))
        )
        self.long_exposure_window_variable = tk.IntVar(
            value=timelapse.DEFAULT_LONG_EXPOSURE_WINDOW
        )
        self.timelapse_estimate_variable = tk.StringVar(value="")
        self.progress_variable = tk.DoubleVar(value=0.0)
        self.operation_status_variable = tk.StringVar(value="")
        self.operation_in_progress = False
        self.render_generation = 0
        self.pending_render: str | None = None
        self._apply_midnight_theme()
        self._build_layout()
        self._apply_dark_titlebar()

    def _apply_dark_titlebar(self) -> None:
        if sys.platform != "win32":
            return
        try:
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            enabled = ctypes.c_int(1)
            for attribute in (20, 19):
                result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(enabled), ctypes.sizeof(enabled)
                )
                if result == 0:
                    break
        except OSError:
            pass

    def _apply_midnight_theme(self) -> None:
        self.root.configure(bg=MIDNIGHT_BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "TFrame", background=MIDNIGHT_BG, borderwidth=0
        )
        style.configure(
            "TLabel", background=MIDNIGHT_BG, foreground=MIDNIGHT_FG
        )
        style.configure(
            "TLabelframe",
            background=MIDNIGHT_BG,
            bordercolor=MIDNIGHT_BORDER,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=MIDNIGHT_BG,
            foreground=MIDNIGHT_ACCENT_ACTIVE,
        )
        style.configure(
            "TButton",
            background=MIDNIGHT_BUTTON_BG,
            foreground=MIDNIGHT_FG,
            bordercolor=MIDNIGHT_BUTTON_BG,
            lightcolor=MIDNIGHT_BUTTON_BG,
            darkcolor=MIDNIGHT_BUTTON_BG,
            focuscolor=MIDNIGHT_BUTTON_BG,
            relief="flat",
            padding=6,
        )
        style.map(
            "TButton",
            background=[("active", MIDNIGHT_BUTTON_HOVER), ("pressed", MIDNIGHT_BUTTON_PRESSED)],
            lightcolor=[("active", MIDNIGHT_BUTTON_HOVER), ("pressed", MIDNIGHT_BUTTON_PRESSED)],
            darkcolor=[("active", MIDNIGHT_BUTTON_HOVER), ("pressed", MIDNIGHT_BUTTON_PRESSED)],
            bordercolor=[("active", MIDNIGHT_BUTTON_HOVER), ("pressed", MIDNIGHT_BUTTON_PRESSED)],
        )
        style.configure(
            "Horizontal.TScale",
            background=MIDNIGHT_BG,
            troughcolor=MIDNIGHT_TROUGH,
            bordercolor=MIDNIGHT_BG,
            lightcolor=MIDNIGHT_ACCENT,
            darkcolor=MIDNIGHT_ACCENT,
        )
        style.configure(
            "TCombobox",
            fieldbackground=MIDNIGHT_PANEL,
            background=MIDNIGHT_PANEL,
            foreground=MIDNIGHT_FG,
            arrowcolor=MIDNIGHT_FG,
            bordercolor=MIDNIGHT_BORDER,
            lightcolor=MIDNIGHT_PANEL,
            darkcolor=MIDNIGHT_PANEL,
            insertcolor=MIDNIGHT_FG,
            padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", MIDNIGHT_PANEL)],
            foreground=[("readonly", MIDNIGHT_FG)],
            selectbackground=[("readonly", MIDNIGHT_PANEL)],
            selectforeground=[("readonly", MIDNIGHT_FG)],
            background=[("active", MIDNIGHT_PANEL), ("readonly", MIDNIGHT_PANEL)],
            bordercolor=[("focus", MIDNIGHT_ACCENT), ("readonly", MIDNIGHT_BORDER)],
            lightcolor=[("focus", MIDNIGHT_PANEL), ("readonly", MIDNIGHT_PANEL)],
            darkcolor=[("focus", MIDNIGHT_PANEL), ("readonly", MIDNIGHT_PANEL)],
            arrowcolor=[("readonly", MIDNIGHT_FG)],
        )
        self.root.option_add("*TCombobox*Listbox.background", MIDNIGHT_PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", MIDNIGHT_FG)
        self.root.option_add(
            "*TCombobox*Listbox.selectBackground", MIDNIGHT_ACCENT
        )
        self.root.option_add(
            "*TCombobox*Listbox.selectForeground", "#e8e9eb"
        )

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

        ttk.Button(left, text="Load Image…", command=self.load_image).pack(fill="x")
        ttk.Button(left, text="Load Folder…", command=self.load_folder).pack(fill="x")
        self.photo_list = tk.Listbox(
            left,
            width=32,
            background=MIDNIGHT_PANEL,
            foreground=MIDNIGHT_FG,
            selectbackground=MIDNIGHT_ACCENT,
            selectforeground="#e8e9eb",
            highlightbackground=MIDNIGHT_BORDER,
            highlightcolor=MIDNIGHT_ACCENT,
            borderwidth=0,
            relief="flat",
        )
        self.photo_list.pack(fill="both", expand=True)
        self.photo_list.bind("<<ListboxSelect>>", self._on_photo_selected)

        preview_toolbar = ttk.Frame(middle)
        preview_toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(preview_toolbar, text="Rotate 90°", command=self.rotate_photo).pack(
            side="left"
        )

        self.preview_label = ttk.Label(middle, anchor="center", background=MIDNIGHT_PANEL)
        self.preview_label.pack(fill="both", expand=True, pady=12)
        self.preview_label.bind("<Configure>", lambda event: self._schedule_preview_refit())

        for group_name, entries in SLIDER_GROUPS.items():
            frame = ttk.LabelFrame(right, text=group_name, padding=8)
            frame.pack(fill="x", pady=10)
            for label, field_name in entries:
                self._add_slider(frame, label, field_name)
        preset_frame = ttk.LabelFrame(right, text="Film Simulation", padding=8)
        preset_frame.pack(fill="x", pady=10)
        preset_box = ttk.Combobox(
            preset_frame,
            textvariable=self.preset_variable,
            values=[NO_PRESET_LABEL, *self.presets],
            state="readonly",
        )
        preset_box.pack(fill="x")
        preset_box.bind("<<ComboboxSelected>>", lambda event: self.schedule_preview())

        self.apply_button = ttk.Button(
            bottom, text="Apply to All Photos", command=self.apply_to_all
        )
        self.apply_button.pack(side="left", padx=4)
        self.export_button = ttk.Button(
            bottom, text="Export Timelapse", command=self.export_timelapse
        )
        self.export_button.pack(side="left", padx=4)

        self.progress_bar = ttk.Progressbar(
            bottom, variable=self.progress_variable, maximum=100.0, length=200
        )
        self.progress_bar.pack(side="right", padx=(4, 8))
        ttk.Label(bottom, textvariable=self.operation_status_variable, width=28).pack(
            side="right", padx=4
        )

        ttk.Label(bottom, text="Timelapse:").pack(side="left", padx=(16, 4))
        mode_box = ttk.Combobox(
            bottom,
            textvariable=self.timelapse_mode_variable,
            values=list(TIMELAPSE_MODE_LABELS),
            state="readonly",
            width=24,
        )
        mode_box.pack(side="left")
        mode_box.bind("<<ComboboxSelected>>", lambda event: self._on_timelapse_mode_changed())

        ttk.Label(bottom, text="Frames/blend:").pack(side="left", padx=(8, 4))
        self.long_exposure_spinbox = ttk.Spinbox(
            bottom,
            from_=2,
            to=120,
            width=4,
            textvariable=self.long_exposure_window_variable,
            command=self._update_timelapse_estimate,
        )
        self.long_exposure_spinbox.pack(side="left")
        self.long_exposure_spinbox.bind(
            "<KeyRelease>", lambda event: self._update_timelapse_estimate()
        )
        self.long_exposure_spinbox.configure(state="disabled")

        ttk.Label(bottom, textvariable=self.timelapse_estimate_variable).pack(
            side="left", padx=(10, 4)
        )
        self._update_timelapse_estimate()

    def _on_timelapse_mode_changed(self) -> None:
        is_long_exposure = (
            TIMELAPSE_MODE_LABELS[self.timelapse_mode_variable.get()]
            == timelapse.MODE_LONG_EXPOSURE
        )
        self.long_exposure_spinbox.configure(
            state="normal" if is_long_exposure else "disabled"
        )
        self._update_timelapse_estimate()

    def _update_timelapse_estimate(self) -> None:
        mode = TIMELAPSE_MODE_LABELS[self.timelapse_mode_variable.get()]
        try:
            window = max(1, self.long_exposure_window_variable.get())
        except tk.TclError:
            window = timelapse.DEFAULT_LONG_EXPOSURE_WINDOW
        count = len(self.photo_paths)
        if count == 0:
            self.timelapse_estimate_variable.set("Load photos to estimate length")
            return
        duration = timelapse.estimate_duration_seconds(
            count, mode, fps=TIMELAPSE_FPS, long_exposure_window=window
        )
        self.timelapse_estimate_variable.set(
            f"~{duration:.1f}s ({count} photo{'s' if count != 1 else ''} @ {TIMELAPSE_FPS}fps)"
        )

    def _add_slider(self, parent: ttk.LabelFrame, label: str, field_name: str) -> None:
        variable = tk.DoubleVar(value=0.0)
        self.slider_variables[field_name] = variable
        lower, upper, decimals = SLIDER_RANGES.get(field_name, DEFAULT_SLIDER_RANGE)
        self.slider_decimals[field_name] = decimals
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=22).pack(side="left")

        def format_value(value: float) -> str:
            text = f"{value:.{decimals}f}"
            return f"+{text}" if value > 0 else text

        value_text = tk.StringVar(value=format_value(0.0))

        def on_change(value: str) -> None:
            snapped = round(float(value), decimals)
            if snapped != variable.get():
                variable.set(snapped)
            value_text.set(format_value(snapped))
            self.schedule_preview()

        scale = ttk.Scale(
            row,
            from_=lower,
            to=upper,
            variable=variable,
            command=on_change,
        )
        scale.pack(side="left", fill="x", expand=True)
        value_label = ttk.Label(row, textvariable=value_text, width=5, anchor="e")
        value_label.pack(side="left", padx=(6, 0))

        def reset(_event: tk.Event) -> str:
            variable.set(0.0)
            value_text.set(format_value(0.0))
            self.schedule_preview()
            return "break"

        scale.bind("<Double-Button-1>", reset)
        value_label.bind("<Double-Button-1>", reset)

    def current_settings(self) -> Settings:
        values = {
            field.name: round(
                self.slider_variables[field.name].get(),
                self.slider_decimals.get(field.name, 0),
            )
            for field in fields(Settings)
            if field.name in self.slider_variables
        }
        return Settings(
            **values,
            rotation=self.rotation,
            film_simulation=self.preset_variable.get(),
        )

    def current_preset(self) -> FilmPreset | None:
        return self.presets.get(self.preset_variable.get())

    def rotate_photo(self) -> None:
        self.rotation = (self.rotation + 90) % 360
        self.schedule_preview()

    def load_image(self) -> None:
        extensions = " ".join(
            f"*{case(ext)}"
            for ext in sorted(utils.SUPPORTED_EXTENSIONS)
            for case in (str.lower, str.upper)
        )
        chosen = filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[("Supported photos", extensions), ("All files", "*.*")],
        )
        if not chosen:
            return
        self.photo_paths = [Path(chosen)]
        self.photo_list.delete(0, tk.END)
        self.photo_list.insert(tk.END, self.photo_paths[0].name)
        self.status_variable.set(f"Loaded {self.photo_paths[0].name}")
        self.photo_list.selection_set(0)
        self._on_photo_selected(None)
        self._update_timelapse_estimate()

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
            self._update_timelapse_estimate()
            return
        self.status_variable.set(f"Loaded {len(self.photo_paths)} photos")
        self.photo_list.selection_set(0)
        self._on_photo_selected(None)
        self._update_timelapse_estimate()

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
        self.last_rendered_image = utils.to_pil_image(image)
        self._refresh_preview_display()

    def _refresh_preview_display(self) -> None:
        if self.last_rendered_image is None:
            return
        box_width = self.preview_label.winfo_width()
        box_height = self.preview_label.winfo_height()
        fitted = fit_within_box(self.last_rendered_image, box_width, box_height)
        self.preview_photo = ImageTk.PhotoImage(fitted)
        self.preview_label.configure(image=self.preview_photo)

    def _schedule_preview_refit(self) -> None:
        if self.pending_resize is not None:
            self.root.after_cancel(self.pending_resize)
        self.pending_resize = self.root.after(
            PREVIEW_DEBOUNCE_MS, self._run_scheduled_refit
        )

    def _run_scheduled_refit(self) -> None:
        self.pending_resize = None
        self._refresh_preview_display()

    def _begin_operation(self, label: str) -> None:
        self.operation_in_progress = True
        self.apply_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.progress_variable.set(0.0)
        self.operation_status_variable.set(label)

    def _make_progress_reporter(self, label: str):
        def report(done: int, total: int) -> None:
            self.root.after(0, lambda: self._apply_progress(done, total, label))

        return report

    def _apply_progress(self, done: int, total: int, label: str) -> None:
        percent = (done / total * 100.0) if total else 0.0
        self.progress_variable.set(percent)
        self.operation_status_variable.set(f"{label} {done}/{total}…")

    def _finish_operation(self, message: str, is_error: bool = False) -> None:
        self.operation_in_progress = False
        self.apply_button.configure(state="normal")
        self.export_button.configure(state="normal")
        self.progress_variable.set(0.0 if is_error else 100.0)
        self.operation_status_variable.set(message)
        if is_error:
            messagebox.showerror("Failed", message)
        else:
            messagebox.showinfo("Finished", message)

    def apply_to_all(self) -> None:
        if self.operation_in_progress:
            return
        if not self.photo_paths:
            messagebox.showinfo("Nothing loaded", "Load a folder of photos first")
            return
        chosen = filedialog.askdirectory(title="Choose an output folder")
        if not chosen:
            return
        output_directory = Path(chosen)
        settings = self.current_settings()
        preset = self.current_preset()
        total = len(self.photo_paths)
        self._begin_operation(f"Processing 0/{total}…")
        report = self._make_progress_reporter("Processing")

        def run() -> None:
            try:
                frames = image_processor.process_files(
                    self.photo_paths,
                    settings,
                    output_directory,
                    preset,
                    progress=report,
                )
            except Exception as error:
                self.root.after(
                    0,
                    lambda: self._finish_operation(
                        f"Batch failed: {error}", is_error=True
                    ),
                )
                return
            self.root.after(
                0,
                lambda: self._finish_operation(
                    f"Saved {len(frames)} photos to {output_directory}"
                ),
            )

        threading.Thread(target=run, daemon=True).start()

    def export_timelapse(self) -> None:
        if self.operation_in_progress:
            return
        if not self.photo_paths:
            messagebox.showinfo("Nothing loaded", "Load photos first")
            return
        if not timelapse.ffmpeg_available():
            messagebox.showerror(
                "ffmpeg required",
                "ffmpeg is required to export a timelapse but was not found on PATH.",
            )
            return
        chosen = filedialog.askdirectory(title="Choose where to save the timelapse")
        if not chosen:
            return
        output_directory = Path(chosen)
        settings = self.current_settings()
        preset = self.current_preset()
        mode = TIMELAPSE_MODE_LABELS[self.timelapse_mode_variable.get()]
        window = max(1, self.long_exposure_window_variable.get())
        total = len(self.photo_paths)
        self._begin_operation(f"Rendering 0/{total}…")
        report = self._make_progress_reporter("Rendering")

        def run() -> None:
            try:
                output = timelapse.build_timelapse(
                    self.photo_paths,
                    output_directory,
                    settings,
                    preset,
                    mode=mode,
                    fps=TIMELAPSE_FPS,
                    long_exposure_window=window,
                    progress=report,
                )
            except Exception as error:
                self.root.after(
                    0,
                    lambda: self._finish_operation(
                        f"Timelapse failed: {error}", is_error=True
                    ),
                )
                return
            self.root.after(
                0, lambda: self._finish_operation(f"Timelapse saved: {output}")
            )

        threading.Thread(target=run, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    PhotoEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
