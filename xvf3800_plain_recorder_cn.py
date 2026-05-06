"""XVF3800 简化中文录音器。

这个版本只做一件事：
1. 把 ReSpeaker XVF3800 固定到指定波束角（默认 30°）
2. 录制该固定波束输出到 16 kHz 单声道 WAV
3. 不做 RMS 门控，不做空间过滤

更适合你现在的考核场景：只验证“30° 方向的声音能被正常录下来”。
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
import wave

import sounddevice as sd

import respeaker_xvf3800_beam_record as recorder

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None


APP_TITLE = "XVF3800 固定波束录音器"


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
        self.geometry("820x560")
        self.minsize(780, 520)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event: threading.Event | None = None
        self.recording = False

        now = time.strftime("%Y%m%d_%H%M%S")
        default_output = pathlib.Path.cwd() / f"xvf3800_plain_{now}.wav"

        self.output_var = tk.StringVar(value=str(default_output))
        self.duration_var = tk.StringVar(value="5")
        self.beam_var = tk.StringVar(value="30")
        self.device_var = tk.StringVar(value="XVF3800")
        self.playback_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪")

        self._build_ui()
        self.after(100, self._drain_log_queue)
        self.bind("<Key>", self._on_keypress)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        settings = ttk.LabelFrame(root, text="录音设置", padding=12)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(2, weight=1)

        ttk.Label(settings, text="输出 WAV 文件").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.output_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(8, 8)
        )
        ttk.Button(settings, text="浏览", command=self._browse_output).grid(
            row=0, column=3, sticky="ew"
        )

        ttk.Label(settings, text="录音时长（秒）").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.duration_var, width=12).grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(settings, text="波束角度（度）").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.beam_var, width=12).grid(
            row=1, column=3, sticky="w", pady=(8, 0)
        )

        ttk.Label(settings, text="设备匹配关键词").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.device_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0)
        )

        ttk.Checkbutton(
            settings,
            text="录完后播放",
            variable=self.playback_var,
        ).grid(row=2, column=3, sticky="w", pady=(8, 0))

        hint = ttk.Label(
            settings,
            text="提示：默认固定 30° 波束。请把声源放在设备前方约 0.5 米处。"
        )
        hint.grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(12, 8))
        self.start_button = ttk.Button(controls, text="开始录音", command=self._start_recording)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            controls, text="停止录音", command=self._stop_recording, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(10, 0))
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

        self._log("就绪。窗口有焦点时，录音过程中按任意键也可停止。")

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存 WAV 文件",
            defaultextension=".wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")],
            initialfile=pathlib.Path(self.output_var.get()).name,
        )
        if path:
            self.output_var.set(path)

    def _validate_float(self, value: str, name: str, min_value: float | None = None) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字") from exc
        if min_value is not None and parsed < min_value:
            raise ValueError(f"{name} 必须大于等于 {min_value}")
        return parsed

    def _normalize_output_path(self, raw_path: str) -> pathlib.Path:
        path = pathlib.Path(raw_path).expanduser()
        if not path.name or str(path).endswith(("\\", "/")) or path.exists() and path.is_dir():
            path = path / f"xvf3800_plain_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        elif path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
        return path.resolve()

    def _start_recording(self) -> None:
        if self.recording:
            return

        try:
            output_path = self._normalize_output_path(self.output_var.get())
            duration = self._validate_float(self.duration_var.get(), "录音时长", 0.1)
            beam = self._validate_float(self.beam_var.get(), "波束角度")
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_var.set(str(output_path))

        self.stop_event = threading.Event()
        self.recording = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("录音中")
        self._log("\n--- 开始录音 ---\n")

        self.worker = threading.Thread(
            target=self._record_worker,
            args=(output_path, duration, beam, self.device_var.get().strip() or None),
            daemon=True,
        )
        self.worker.start()

    def _stop_recording(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

    def _on_keypress(self, _event) -> None:
        if self.recording and self.stop_event is not None:
            self.stop_event.set()

    def _record_worker(
        self,
        output_path: pathlib.Path,
        duration: float,
        beam: float,
        device_hint: str | None,
    ) -> None:
        ctrl = None
        try:
            writer = TextQueueWriter(self.log_queue)
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                dev = recorder.find_device()
                if dev is None:
                    raise RuntimeError(
                        "没有找到 XVF3800 USB 控制设备。\n"
                        "请确认设备已连接，并且控制接口已切到 WinUSB/libusb。"
                    )

                ctrl = recorder.ReSpeakerControl(dev)
                recorder.configure_device(ctrl, beam)

                input_index = recorder.pick_input_device(device_hint)
                if input_index is None:
                    devices = sd.query_devices()
                    details = "\n".join(
                        f"  [{i}] {d['name']} (inputs={d['max_input_channels']}, "
                        f"sr={int(d['default_samplerate'])} Hz)"
                        for i, d in enumerate(devices)
                        if d["max_input_channels"] > 0
                    )
                    raise RuntimeError(
                        "没有找到匹配的音频输入设备。\n\n"
                        f"可用输入设备：\n{details}"
                    )

                dev_info = sd.query_devices(input_index)
                print(f"音频输入：[{input_index}] {dev_info['name']}")
                print(f"采样率：{recorder.SAMPLE_RATE} Hz | 声道：{recorder.CHANNELS}（立体声输入，保存为单声道）")
                print(f"输出文件：{output_path}")
                print(f"波束角度：{beam:.1f}°")
                print(f"录音时长：{duration:.1f} 秒")
                print("按窗口中的任意键或点击“停止录音”可提前结束。\n")

                q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)
                peak_rms = 0.0
                received_frames = 0
                t_start = time.monotonic()
                last_status_time = t_start

                def callback(indata, _frames, _time_info, status) -> None:
                    try:
                        q.put_nowait(bytes(indata))
                    except queue.Full:
                        pass

                with wave.open(str(output_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(recorder.SAMPLE_WIDTH_BYTES)
                    wf.setframerate(recorder.SAMPLE_RATE)

                    with sd.RawInputStream(
                        samplerate=recorder.SAMPLE_RATE,
                        channels=recorder.CHANNELS,
                        dtype="int16",
                        blocksize=recorder.BLOCKSIZE,
                        device=input_index,
                        callback=callback,
                    ):
                        while True:
                            if self.stop_event is not None and self.stop_event.is_set():
                                print("已手动停止录音。")
                                break

                            elapsed = time.monotonic() - t_start
                            if elapsed >= duration:
                                print(f"已达到设定时长（{duration:.1f} 秒）。")
                                break

                            try:
                                chunk = q.get(timeout=0.1)
                            except queue.Empty:
                                continue

                            received_frames += len(chunk) // (recorder.SAMPLE_WIDTH_BYTES * recorder.CHANNELS)
                            level = recorder.rms_int16_mono(chunk, channels=recorder.CHANNELS)
                            peak_rms = max(peak_rms, level)

                            mono = recorder.extract_mono_bytes(chunk, channels=recorder.CHANNELS)
                            wf.writeframesraw(mono)

                            now = time.monotonic()
                            if now - last_status_time >= 2.0:
                                print(
                                    f"  [{elapsed:5.1f}s] 当前 RMS={level:.4f}  Peak RMS={peak_rms:.4f}  "
                                    f"已保存={received_frames / recorder.SAMPLE_RATE:.1f}s"
                                )
                                last_status_time = now

                captured_sec = received_frames / recorder.SAMPLE_RATE

                print("\n--- 录音完成 ---")
                print(f"输出文件：{output_path}")
                print(f"采集时长：{captured_sec:.2f} 秒")
                print(f"Peak RMS：{peak_rms:.4f}")
                print(f"波束角度：{beam:.1f}°")
                print("说明：本版本不做门控，WAV 保存的是固定 30° 波束的完整录音。")

                if self.playback_var.get():
                    playback_status = self._playback(output_path)
                    print(f"播放状态：{playback_status}")

            self.log_queue.put("__STATUS__DONE")
        except Exception as exc:
            self.log_queue.put(f"\n错误：{exc}\n")
            self.log_queue.put("__STATUS__ERROR")
        finally:
            if ctrl is not None:
                with contextlib.suppress(Exception):
                    ctrl.close()

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
                    self.recording = False
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                elif item == "__STATUS__ERROR":
                    self.status_var.set("出错")
                    self.recording = False
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
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
