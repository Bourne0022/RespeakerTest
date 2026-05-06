"""Small Windows GUI for ReSpeaker XVF3800 beamformed recording.

The GUI reuses `respeaker_xvf3800_beam_record.record()` directly.  It does not
modify the recorder module or the generated WAV files.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import respeaker_xvf3800_beam_record as recorder

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None


APP_TITLE = "XVF3800 Beam Recorder"


class TextQueueWriter(io.TextIOBase):
    def __init__(self, out_queue: "queue.Queue[str]") -> None:
        self.out_queue = out_queue

    def write(self, text: str) -> int:
        if text:
            self.out_queue.put(text)
        return len(text)

    def flush(self) -> None:
        pass


class RecorderApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("760x560")
        self.minsize(720, 500)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None

        now = time.strftime("%Y%m%d_%H%M%S")
        default_output = pathlib.Path.cwd() / f"xvf3800_record_{now}.wav"

        self.output_var = tk.StringVar(value=str(default_output))
        self.duration_var = tk.StringVar(value="5")
        self.beam_var = tk.StringVar(value="30")
        self.rms_var = tk.StringVar(value="0.03")
        self.attack_var = tk.StringVar(value="40")
        self.hold_var = tk.StringVar(value="400")
        self.device_var = tk.StringVar(value="XVF3800")
        self.spatial_var = tk.BooleanVar(value=False)
        self.calibration_var = tk.StringVar(value=str(pathlib.Path.cwd() / "calibration.json"))
        self.playback_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        settings = ttk.LabelFrame(root, text="Recording Settings", padding=12)
        settings.pack(fill="x")

        for col in range(4):
            settings.columnconfigure(col, weight=1)

        ttk.Label(settings, text="Output WAV").grid(row=0, column=0, sticky="w")
        out_entry = ttk.Entry(settings, textvariable=self.output_var)
        out_entry.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 8))
        ttk.Button(settings, text="Browse", command=self._browse_output).grid(row=0, column=3, sticky="ew")

        ttk.Label(settings, text="Duration (s)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.duration_var, width=10).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="Beam angle (deg)").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.beam_var, width=10).grid(row=1, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="RMS threshold").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.rms_var, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="Device hint").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.device_var, width=18).grid(row=2, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="Attack ms").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.attack_var, width=10).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="Hold ms").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.hold_var, width=10).grid(row=3, column=3, sticky="w", pady=(8, 0))

        options = ttk.Frame(settings)
        options.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(options, text="Enable spatial filter", variable=self.spatial_var).pack(side="left")
        ttk.Checkbutton(options, text="Play after recording", variable=self.playback_var).pack(side="left", padx=(18, 0))

        ttk.Label(settings, text="Calibration JSON").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.calibration_var).grid(row=5, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(settings, text="Browse", command=self._browse_calibration).grid(row=5, column=3, sticky="ew", pady=(8, 0))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(12, 8))
        self.start_button = ttk.Button(controls, text="Start Recording", command=self._start_recording)
        self.start_button.pack(side="left")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        log_frame = ttk.LabelFrame(root, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", height=16)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

        self._log("Ready. Default mode records beamformed audio without spatial filtering.\n")

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save WAV",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile=pathlib.Path(self.output_var.get()).name,
        )
        if path:
            self.output_var.set(path)

    def _browse_calibration(self) -> None:
        path = filedialog.askopenfilename(
            title="Select calibration JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.calibration_var.set(path)

    def _validate_float(self, value: str, name: str, min_value: float | None = None) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
        if min_value is not None and parsed < min_value:
            raise ValueError(f"{name} must be >= {min_value}")
        return parsed

    def _start_recording(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo(APP_TITLE, "Recording is already running.")
            return

        try:
            output_path = pathlib.Path(self.output_var.get()).expanduser().resolve()
            if output_path.exists() and output_path.is_dir():
                output_path = output_path / f"xvf3800_record_{time.strftime('%Y%m%d_%H%M%S')}.wav"
                self.output_var.set(str(output_path))
            elif output_path.suffix.lower() != ".wav":
                output_path = output_path.with_suffix(".wav")
                self.output_var.set(str(output_path))
            duration = self._validate_float(self.duration_var.get(), "Duration", 0.1)
            beam = self._validate_float(self.beam_var.get(), "Beam angle")
            rms = self._validate_float(self.rms_var.get(), "RMS threshold", 0.0)
            attack = self._validate_float(self.attack_var.get(), "Attack ms", 0.0)
            hold = self._validate_float(self.hold_var.get(), "Hold ms", 0.0)
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.start_button.configure(state="disabled")
        self.status_var.set("Recording...")
        self._log("\n--- Recording started ---\n")

        self.worker = threading.Thread(
            target=self._record_worker,
            args=(output_path, duration, beam, rms, attack, hold),
            daemon=True,
        )
        self.worker.start()

    def _record_worker(
        self,
        output_path: pathlib.Path,
        duration: float,
        beam: float,
        rms: float,
        attack: float,
        hold: float,
    ) -> None:
        ref_energy = None
        enable_spatial = self.spatial_var.get()

        try:
            if enable_spatial:
                cal_path = pathlib.Path(self.calibration_var.get()).expanduser().resolve()
                if cal_path.exists():
                    cal = recorder._load_calibration(str(cal_path))
                    ref_energy = cal.get("ref_energy")
                    self.log_queue.put(f"Loaded calibration: {cal_path}\n")
                else:
                    self.log_queue.put("Calibration JSON not found; spatial energy gate disabled.\n")

            writer = TextQueueWriter(self.log_queue)
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                stats = recorder.record(
                    output_path=str(output_path),
                    duration_sec=duration,
                    rms_threshold=rms,
                    device_hint=self.device_var.get().strip() or None,
                    beam_deg=beam,
                    attack_ms=attack,
                    hold_ms=hold,
                    ref_energy=ref_energy,
                    enable_spatial=enable_spatial,
                )

            captured_sec = stats.received_frames / recorder.SAMPLE_RATE
            saved_sec = stats.saved_frames / recorder.SAMPLE_RATE
            self.log_queue.put("\n--- Recording finished ---\n")
            self.log_queue.put(f"Output: {output_path}\n")
            self.log_queue.put(f"Captured: {captured_sec:.2f} s\n")
            self.log_queue.put(f"Saved: {saved_sec:.2f} s\n")
            self.log_queue.put(f"Peak RMS: {stats.peak_rms:.4f}\n")
            if stats.doa_rejects or stats.energy_rejects:
                self.log_queue.put(f"DOA rejects: {stats.doa_rejects}\n")
                self.log_queue.put(f"Energy rejects: {stats.energy_rejects}\n")

            if self.playback_var.get():
                status = self._playback(output_path)
                self.log_queue.put(f"Playback: {status}\n")

            self.log_queue.put("__STATUS__DONE")
        except Exception as exc:
            self.log_queue.put(f"\nERROR: {exc}\n")
            self.log_queue.put("__STATUS__ERROR")

    def _playback(self, path: pathlib.Path) -> str:
        if winsound is None:
            return "SKIPPED"
        try:
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return "OK"
        except Exception as exc:
            return f"FAILED: {exc}"

    def _drain_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__STATUS__DONE":
                    self.status_var.set("Done")
                    self.start_button.configure(state="normal")
                elif item == "__STATUS__ERROR":
                    self.status_var.set("Error")
                    self.start_button.configure(state="normal")
                else:
                    self._log(item)
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")


def main() -> int:
    app = RecorderApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
