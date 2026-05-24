import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "Data"
PHOTOS_ROOT = DATA_ROOT / "Photos"
MODELS_DIR = PROJECT_DIR / "models"
FOLDER_RE = re.compile(r"^[A-Za-z0-9А-Яа-яІіЇїЄєҐґ _-]+$")


class JointInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sign Language Project Interface")
        self.geometry("760x520")
        self.minsize(680, 460)
        self.current_process = None

        DATA_ROOT.mkdir(exist_ok=True)
        PHOTOS_ROOT.mkdir(parents=True, exist_ok=True)

        self.dataset_name = tk.StringVar()
        self.classes = tk.IntVar(value=29)
        self.dataset_size = tk.IntVar(value=100)
        self.model_type = tk.StringVar(value="MLP")
        self.person_model = tk.StringVar(value="Combined")
        self.train_epochs = tk.IntVar(value=30)
        self.status = tk.StringVar(value="Ready")

        self._build_ui()
        self._refresh_datasets()

    def _build_ui(self):
        main = ttk.Frame(self, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        settings = ttk.LabelFrame(main, text="Dataset", padding=12)
        settings.pack(fill=tk.X)

        ttk.Label(settings, text="Folder name").grid(row=0, column=0, sticky=tk.W)
        self.dataset_combo = ttk.Combobox(settings, textvariable=self.dataset_name)
        self.dataset_combo.grid(row=0, column=1, sticky=tk.EW, padx=(8, 8))
        ttk.Button(settings, text="Refresh", command=self._refresh_datasets).grid(row=0, column=2)

        ttk.Label(settings, text="Classes").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Spinbox(settings, from_=1, to=100, textvariable=self.classes, width=8).grid(
            row=1, column=1, sticky=tk.W, padx=(8, 0), pady=(10, 0)
        )

        ttk.Label(settings, text="Images per class").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Spinbox(settings, from_=1, to=10000, textvariable=self.dataset_size, width=8).grid(
            row=2, column=1, sticky=tk.W, padx=(8, 0), pady=(10, 0)
        )

        settings.columnconfigure(1, weight=1)

        recognition = ttk.LabelFrame(main, text="Real-Time Recognition Model", padding=12)
        recognition.pack(fill=tk.X, pady=(12, 0))

        ttk.Label(recognition, text="Model type").grid(row=0, column=0, sticky=tk.W)
        self.model_type_combo = ttk.Combobox(
            recognition,
            textvariable=self.model_type,
            values=("MLP", "CNN"),
            state="readonly",
            width=12,
        )
        self.model_type_combo.grid(row=0, column=1, sticky=tk.W, padx=(8, 20))
        self.model_type_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_model_controls())

        ttk.Label(recognition, text="Person").grid(row=0, column=2, sticky=tk.W)
        self.person_combo = ttk.Combobox(
            recognition,
            textvariable=self.person_model,
            values=("K", "O", "R", "Combined"),
            state="readonly",
            width=12,
        )
        self.person_combo.grid(row=0, column=3, sticky=tk.W, padx=(8, 0))
        ttk.Label(recognition, text="Epochs").grid(row=0, column=4, sticky=tk.W, padx=(20, 0))
        ttk.Spinbox(recognition, from_=1, to=500, textvariable=self.train_epochs, width=8).grid(
            row=0, column=5, sticky=tk.W, padx=(8, 0)
        )
        recognition.columnconfigure(6, weight=1)

        actions = ttk.Frame(main)
        actions.pack(fill=tk.X, pady=14)

        ttk.Button(actions, text="Create Dataset", command=self.create_dataset).pack(side=tk.LEFT)
        ttk.Button(actions, text="Extract Keypoints", command=self.extract_keypoints).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Train Model", command=self.train_model).pack(side=tk.LEFT)
        ttk.Button(actions, text="Real-Time Recognition", command=self.real_time_recognition).pack(side=tk.LEFT)
        ttk.Button(actions, text="Stop Process", command=self.stop_process).pack(side=tk.RIGHT)

        path_frame = ttk.Frame(main)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(path_frame, text="Path:").pack(side=tk.LEFT)
        self.path_label = ttk.Label(path_frame, text=str(PHOTOS_ROOT), foreground="#555555")
        self.path_label.pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

        ttk.Label(main, textvariable=self.status).pack(anchor=tk.W)

        log_frame = ttk.LabelFrame(main, text="Output", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log = tk.Text(log_frame, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.configure(yscrollcommand=scrollbar.set)

        self._update_model_controls()

    def _refresh_datasets(self):
        folders = sorted(p.name for p in PHOTOS_ROOT.iterdir() if p.is_dir())
        self.dataset_combo["values"] = folders
        if not self.dataset_name.get() and folders:
            self.dataset_name.set(folders[0])
        self._update_path_label()

    def _update_path_label(self):
        name = self.dataset_name.get().strip()
        self.path_label.configure(text=str(PHOTOS_ROOT / name) if name else str(PHOTOS_ROOT / "<folder name>"))

    def _update_model_controls(self):
        is_mlp = self.model_type.get() == "MLP"
        current = self.person_model.get()
        if is_mlp:
            values = ("K", "O", "R", "Combined", "Custom")
            if current not in values:
                self.person_model.set("Combined")
        else:
            values = ("General", "Custom")
            if current not in values:
                self.person_model.set("General")
        self.person_combo.configure(values=values, state="readonly")

    def _dataset_path(self, create=False):
        name = self.dataset_name.get().strip()
        if not name:
            raise ValueError("Enter dataset folder name.")
        if not FOLDER_RE.match(name) or os.path.basename(name) != name:
            raise ValueError("Use only letters, numbers, spaces, underscores, or hyphens in the folder name.")

        path = PHOTOS_ROOT / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_script(self, script_name, *args):
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Process is running", "Stop the current process before starting another one.")
            return

        command = [sys.executable, str(PROJECT_DIR / script_name), *map(str, args)]
        self._append_log(f"\n> {' '.join(command)}\n")
        self.status.set(f"Running {script_name}...")

        thread = threading.Thread(target=self._process_worker, args=(command,), daemon=True)
        thread.start()

    def _process_worker(self, command):
        self.current_process = subprocess.Popen(
            command,
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in self.current_process.stdout:
            self.after(0, self._append_log, line)

        return_code = self.current_process.wait()
        self.after(0, self._process_finished, return_code)

    def _process_finished(self, return_code):
        self.status.set("Ready" if return_code == 0 else f"Finished with code {return_code}")
        self.current_process = None
        self._refresh_datasets()

    def _append_log(self, text):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _show_error(self, error):
        messagebox.showerror("Input error", str(error))

    def create_dataset(self):
        try:
            path = self._dataset_path(create=True)
            self._update_path_label()
            self._run_script(
                "get_photos.py",
                "--data-dir",
                path,
                "--classes",
                self.classes.get(),
                "--dataset-size",
                self.dataset_size.get(),
            )
        except Exception as error:
            self._show_error(error)

    def extract_keypoints(self):
        try:
            path = self._dataset_path(create=False)
            if not path.exists():
                raise FileNotFoundError(f"Dataset folder does not exist: {path}")
            output_path = DATA_ROOT / f"{path.name}_keypoints.pickle"
            self._update_path_label()
            self._run_script("extract_keypoints.py", "--data-dir", path, "--output", output_path)
        except Exception as error:
            self._show_error(error)

    def _safe_model_name(self, value):
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
        return cleaned.strip("_") or "custom"

    def _custom_model_paths(self, model_type):
        path = self._dataset_path(create=False)
        model_name = self._safe_model_name(path.name)
        prefix = MODELS_DIR / f"custom_{model_name}"
        suffix = model_type.lower()
        return {
            "name": model_name,
            "model": prefix.with_name(f"{prefix.name}_{suffix}.h5"),
            "labels": prefix.with_name(f"{prefix.name}_{suffix}_label_map.pkl"),
        }

    def train_model(self):
        try:
            path = self._dataset_path(create=False)
            if not path.exists():
                raise FileNotFoundError(f"Dataset folder does not exist: {path}")

            args = [
                "--data-dir",
                path,
                "--model-type",
                self.model_type.get(),
                "--name",
                self._safe_model_name(path.name),
                "--epochs",
                self.train_epochs.get(),
            ]

            if self.model_type.get() == "MLP":
                keypoints_path = DATA_ROOT / f"{path.name}_keypoints.pickle"
                if not keypoints_path.exists():
                    raise FileNotFoundError(
                        f"Keypoints file does not exist: {keypoints_path}. Run Extract Keypoints first."
                    )
                args.extend(["--keypoints", keypoints_path])

            self._run_script("train_models.py", *args)
        except Exception as error:
            self._show_error(error)

    def real_time_recognition(self):
        try:
            args = ["--model-type", self.model_type.get()]
            if self.model_type.get() == "MLP":
                args.extend(["--person", self.person_model.get()])
                if self.person_model.get() == "Custom":
                    custom = self._custom_model_paths("MLP")
                    if not custom["model"].exists() or not custom["labels"].exists():
                        raise FileNotFoundError("Train an MLP custom model for this dataset first.")
                    args.extend(
                        [
                            "--model-name",
                            custom["name"],
                            "--mlp-model-path",
                            custom["model"],
                            "--mlp-labels-path",
                            custom["labels"],
                        ]
                    )
            else:
                if self.person_model.get() == "Custom":
                    custom = self._custom_model_paths("CNN")
                    if not custom["model"].exists() or not custom["labels"].exists():
                        raise FileNotFoundError("Train a CNN custom model for this dataset first.")
                    args.extend(
                        [
                            "--model-name",
                            custom["name"],
                            "--cnn-model-path",
                            custom["model"],
                            "--cnn-labels-path",
                            custom["labels"],
                        ]
                    )

            self._run_script("real_time_recogntion.py", *args)
        except Exception as error:
            self._show_error(error)

    def stop_process(self):
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.status.set("Stopping process...")
        else:
            self.status.set("No process is running")


if __name__ == "__main__":
    app = JointInterface()
    app.mainloop()
