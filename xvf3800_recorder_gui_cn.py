"""ReSpeaker XVF3800 中文录音界面."""

from __future__ import annotations

import contextlib
import io
import pathlib
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import respeaker_xvf3800_beam_record as recorder

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None


APP_TITLE = "XVF3800 波束录音工具"
DEFAULT_REF_RMS = 0.086293


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
        self.geometry("940x600")
        self.minsize(860, 540)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.auto_output_path = True

        self.output_var = tk.StringVar(value=str(self._default_output_path()))
        self.duration_var = tk.StringVar(value="0")
        self.beam_var = tk.StringVar(value="30")
        self.rms_var = tk.StringVar(value="0.03")
        self.attack_var = tk.StringVar(value="40")
        self.hold_var = tk.StringVar(value="400")
        self.device_var = tk.StringVar(value="XVF3800")
        self.spatial_var = tk.BooleanVar(value=True)
        self.distance_gate_var = tk.BooleanVar(value=True)
        self.trigger_var = tk.BooleanVar(value=True)
        self.denoise_var = tk.BooleanVar(value=True)
        self.calibration_var = tk.StringVar(value=str(self._app_dir() / "calibration.json"))
        self.playback_var = tk.BooleanVar(value=False)
        self.angle_tolerance_var = tk.StringVar(value="25")
        self.ratio_threshold_var = tk.StringVar(value="0.0")
        self.status_var = tk.StringVar(value="就绪")

        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _default_output_path(self) -> pathlib.Path:
        now = time.strftime("%Y%m%d_%H%M%S")
        return self._app_dir() / f"xvf3800_record_{now}.wav"

    def _app_dir(self) -> pathlib.Path:
        if getattr(sys, "frozen", False):
            return pathlib.Path(sys.executable).resolve().parent
        return pathlib.Path.cwd()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        settings = ttk.LabelFrame(root, text="录音设置", padding=12)
        settings.pack(fill="x")

        for col in range(4):
            settings.columnconfigure(col, weight=1)

        ttk.Label(settings, text="输出 WAV 文件").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.output_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 8))
        ttk.Button(settings, text="浏览", command=self._browse_output).grid(row=0, column=3, sticky="ew")

        ttk.Label(settings, text="录音时长（秒）").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.duration_var, width=10).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="波束角度（度）").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.beam_var, width=10).grid(row=1, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="RMS 门限").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.rms_var, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="设备匹配关键词").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.device_var, width=18).grid(row=2, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="Attack 时间（ms）").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.attack_var, width=10).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="Hold 时间（ms）").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.hold_var, width=10).grid(row=3, column=3, sticky="w", pady=(8, 0))

        options = ttk.Frame(settings)
        options.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(options, text="启用角度门控（无需定标）", variable=self.spatial_var).pack(side="left")
        ttk.Checkbutton(options, text="启用距离门控（RMS 标定）", variable=self.distance_gate_var).pack(side="left", padx=(18, 0))
        ttk.Checkbutton(options, text="检测到人声后才生成文件", variable=self.trigger_var).pack(side="left", padx=(18, 0))
        ttk.Checkbutton(options, text="启用轻量降噪", variable=self.denoise_var).pack(side="left", padx=(18, 0))
        ttk.Checkbutton(options, text="录完后播放", variable=self.playback_var).pack(side="left", padx=(18, 0))

        ttk.Label(settings, text="角度容差（度）").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.angle_tolerance_var, width=10).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="波束比阈值（0关闭）").grid(row=5, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.ratio_threshold_var, width=10).grid(row=5, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="标定文件（仅距离门控使用）").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.calibration_var).grid(row=6, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(settings, text="浏览", command=self._browse_calibration).grid(row=6, column=3, sticky="ew", pady=(8, 0))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(12, 8))
        self.start_button = ttk.Button(controls, text="开始录音", command=self._start_recording)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="停止录音", command=self._stop_recording, state="disabled")
        self.stop_button.pack(side="left", padx=(10, 0))
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        hint = ttk.Label(
            root,
            text="提示：当前默认按设备语音检测 + 30° 方向门控录音；距离门控使用标定里的 ref_rms。",
            foreground="#444",
        )
        hint.pack(fill="x", pady=(0, 8))

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", height=16)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

        self._log("就绪。默认启用 30° 人声/DOA 门控和 RMS 距离门控；波束比阈值默认关闭。\n")

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存 WAV 文件",
            defaultextension=".wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")],
            initialfile=pathlib.Path(self.output_var.get()).name,
        )
        if path:
            self.auto_output_path = False
            self.output_var.set(path)

    def _browse_calibration(self) -> None:
        path = filedialog.askopenfilename(
            title="选择标定 JSON 文件",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if path:
            self.calibration_var.set(path)

    def _validate_float(self, value: str, name: str, min_value: float | None = None) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字") from exc
        if min_value is not None and parsed < min_value:
            raise ValueError(f"{name} 必须大于等于 {min_value}")
        return parsed

    def _start_recording(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo(APP_TITLE, "正在录音，请等待当前任务结束。")
            return

        try:
            if self.auto_output_path:
                self.output_var.set(str(self._default_output_path()))
            output_path = pathlib.Path(self.output_var.get()).expanduser().resolve()
            if output_path.exists() and output_path.is_dir():
                output_path = output_path / f"xvf3800_record_{time.strftime('%Y%m%d_%H%M%S')}.wav"
                self.output_var.set(str(output_path))
            elif output_path.suffix.lower() != ".wav":
                output_path = output_path.with_suffix(".wav")
                self.output_var.set(str(output_path))
            duration = self._validate_float(self.duration_var.get(), "录音时长", 0.0)
            beam = self._validate_float(self.beam_var.get(), "波束角度")
            rms = self._validate_float(self.rms_var.get(), "RMS 门限", 0.0)
            attack = self._validate_float(self.attack_var.get(), "Attack 时间", 0.0)
            hold = self._validate_float(self.hold_var.get(), "Hold 时间", 0.0)
            angle_tolerance = self._validate_float(self.angle_tolerance_var.get(), "角度容差", 0.0)
            ratio_threshold = self._validate_float(self.ratio_threshold_var.get(), "波束比阈值", 0.0)
            if ratio_threshold > 1.0:
                raise ValueError("波束比阈值必须小于等于 1.0")
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("录音中...")
        self._log("\n--- 开始录音 ---\n")

        self.worker = threading.Thread(
            target=self._record_worker,
            args=(output_path, duration, beam, rms, attack, hold, angle_tolerance, ratio_threshold),
            daemon=True,
        )
        self.worker.start()

    def _stop_recording(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("正在停止...")

    def _record_worker(
        self,
        output_path: pathlib.Path,
        duration: float,
        beam: float,
        rms: float,
        attack: float,
        hold: float,
        angle_tolerance: float,
        ratio_threshold: float,
    ) -> None:
        ref_rms = None
        enable_spatial = self.spatial_var.get()
        enable_distance = self.distance_gate_var.get()

        try:
            if enable_spatial and enable_distance:
                cal_path = pathlib.Path(self.calibration_var.get()).expanduser().resolve()
                if cal_path.exists():
                    cal = recorder._load_calibration(str(cal_path))
                    ref_rms = cal.get("ref_rms", cal.get("ref_energy"))
                    if ref_rms is None or float(ref_rms) <= 0:
                        ref_rms = DEFAULT_REF_RMS
                        self.log_queue.put(
                            f"标定文件无有效 ref_rms，改用默认参考值 {DEFAULT_REF_RMS:.6f}\n"
                        )
                    else:
                        ref_rms = float(ref_rms)
                        self.log_queue.put(f"已加载标定文件：{cal_path}\n")
                else:
                    ref_rms = DEFAULT_REF_RMS
                    self.log_queue.put(
                        f"未找到标定文件：使用默认参考值 {DEFAULT_REF_RMS:.6f}。\n"
                    )
            elif enable_spatial and not enable_distance:
                self.log_queue.put("未启用距离门控：仅使用 30° 角度门控 + 人声触发。\n")

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
                    angle_tolerance_deg=angle_tolerance,
                    ref_rms=ref_rms,
                    distance_ratio=0.30,
                    ratio_threshold=ratio_threshold,
                    enable_spatial=enable_spatial,
                    stop_event=self.stop_event,
                    trigger_on_voice=self.trigger_var.get(),
                    denoise=self.denoise_var.get(),
                )

            captured_sec = stats.received_frames / recorder.SAMPLE_RATE
            saved_sec = stats.saved_frames / recorder.SAMPLE_RATE
            self.log_queue.put("\n--- 录音完成 ---\n")
            if output_path.exists():
                self.log_queue.put(f"输出文件：{output_path}\n")
            else:
                self.log_queue.put("未检测到符合 30° 方向的人声，未生成 WAV 文件。\n")
            self.log_queue.put(f"采集时长：{captured_sec:.2f} 秒\n")
            self.log_queue.put(f"保存时长：{saved_sec:.2f} 秒\n")
            self.log_queue.put(f"Peak RMS：{stats.peak_rms:.4f}\n")
            if stats.doa_samples:
                self.log_queue.put(
                    f"DOA 平均值：{sum(stats.doa_samples) / len(stats.doa_samples):.1f}°，"
                    f"样本数：{len(stats.doa_samples)}\n"
                )
            if stats.doa_rejects or stats.energy_rejects:
                self.log_queue.put(f"DOA 拒绝块数：{stats.doa_rejects}\n")
                self.log_queue.put(f"距离/RMS 拒绝块数：{stats.energy_rejects}\n")
                self.log_queue.put(f"波束比拒绝块数：{stats.ratio_rejects}\n")
            if stats.denoise_applied:
                self.log_queue.put(
                    f"轻量降噪：已应用，噪声参考 RMS={stats.denoise_noise_rms:.4f}，"
                    f"处理后 Peak={stats.denoise_peak_rms:.4f}\n"
                )

            if self.playback_var.get() and output_path.exists():
                status = self._playback(output_path)
                self.log_queue.put(f"播放状态：{status}\n")

            self.log_queue.put("__STATUS__DONE")
        except Exception as exc:
            self.log_queue.put(f"\n错误：{exc}\n")
            self.log_queue.put("__STATUS__ERROR")

    def _playback(self, path: pathlib.Path) -> str:
        if winsound is None:
            return "跳过：当前环境不支持 winsound"
        try:
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return "正常"
        except Exception as exc:
            return f"失败：{exc}"

    def _drain_log_queue(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item == "__STATUS__DONE":
                    self.status_var.set("完成")
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.stop_event = None
                elif item == "__STATUS__ERROR":
                    self.status_var.set("出错")
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.stop_event = None
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
