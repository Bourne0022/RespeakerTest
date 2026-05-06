"""XVF3800 中文标定工具。

用途：
1. 在 30° 波束、约 0.5 米位置录制若干次短音频
2. 自动生成每次 WAV
3. 汇总 Peak RMS 到 CSV
4. 生成 calibration.json，供录音工具里的“空间过滤”使用

说明：
这里的“标定”是把当前环境下、目标位置的参考响度记录下来，
不是测真实距离。空间过滤依赖这个参考值做近似判断。
"""

from __future__ import annotations

import contextlib
import csv
import io
import pathlib
import queue
import statistics
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
import wave

import respeaker_xvf3800_beam_record as recorder

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None


APP_TITLE = "XVF3800 标定工具"


class TextQueueWriter(io.TextIOBase):
    def __init__(self, out_queue: "queue.Queue[str]") -> None:
        self.out_queue = out_queue

    def write(self, text: str) -> int:
        if text:
            self.out_queue.put(text)
        return len(text)

    def flush(self) -> None:
        pass


@dataclass
class RunResult:
    index: int
    filename: str
    wav_path: pathlib.Path
    peak_rms: float
    wav_duration_sec: float
    beam_deg: float
    note: str


class CalibrationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("860x620")
        self.minsize(820, 560)

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event: threading.Event | None = None

        now = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = pathlib.Path.cwd()
        self.csv_var = tk.StringVar(value=str(self.output_dir / "calibration_30deg_0.5m.csv"))
        self.json_var = tk.StringVar(value=str(self.output_dir / "calibration.json"))
        self.count_var = tk.StringVar(value="5")
        self.duration_var = tk.StringVar(value="5")
        self.beam_var = tk.StringVar(value="30")
        self.distance_var = tk.StringVar(value="0.5")
        self.rms_var = tk.StringVar(value="0.03")
        self.attack_var = tk.StringVar(value="40")
        self.hold_var = tk.StringVar(value="400")
        self.device_var = tk.StringVar(value="XVF3800")
        self.playback_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪")
        self.note_var = tk.StringVar(value="距离约 0.5m")
        self.prefix_var = tk.StringVar(value=f"calibration_30deg_0p5m")

        self._build_ui()
        self.after(100, self._drain_log_queue)
        self.bind("<Key>", self._on_keypress)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        settings = ttk.LabelFrame(root, text="标定设置", padding=12)
        settings.pack(fill="x")
        for col in range(4):
            settings.columnconfigure(col, weight=1)

        ttk.Label(settings, text="输出 CSV 文件").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.csv_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(8, 8))
        ttk.Button(settings, text="浏览", command=self._browse_csv).grid(row=0, column=3, sticky="ew")

        ttk.Label(settings, text="输出 JSON 文件").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.json_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Button(settings, text="浏览", command=self._browse_json).grid(row=1, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(settings, text="标定次数").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.count_var, width=10).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="每次时长（秒）").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.duration_var, width=10).grid(row=2, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="波束角度（度）").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.beam_var, width=10).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="目标距离（米）").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.distance_var, width=10).grid(row=3, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="RMS 门限").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.rms_var, width=10).grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="设备匹配关键词").grid(row=4, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.device_var).grid(row=4, column=3, sticky="ew", pady=(8, 0))

        ttk.Label(settings, text="Attack 时间（ms）").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.attack_var, width=10).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(settings, text="Hold 时间（ms）").grid(row=5, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.hold_var, width=10).grid(row=5, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="文件前缀").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.prefix_var).grid(row=6, column=1, columnspan=2, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Checkbutton(settings, text="录完后播放", variable=self.playback_var).grid(row=6, column=3, sticky="w", pady=(8, 0))

        ttk.Label(settings, text="备注").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.note_var).grid(row=7, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(8, 0))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(12, 8))
        self.start_button = ttk.Button(controls, text="开始标定", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="停止标定", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(10, 0))
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=(16, 0))

        hint = ttk.Label(
            root,
            text="提示：把声源放在设备前方约 0.5 米、30° 方向。标定完成后，会生成 CSV 和 calibration.json。",
            foreground="#444",
        )
        hint.pack(fill="x", pady=(0, 8))

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word", height=18)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scroll.set)

        self._log("就绪。按“开始标定”后，会按设定次数录制并生成 calibration.json。")

    def _browse_csv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存 CSV 文件",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
            initialfile=pathlib.Path(self.csv_var.get()).name,
        )
        if path:
            self.csv_var.set(path)

    def _browse_json(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存 JSON 文件",
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            initialfile=pathlib.Path(self.json_var.get()).name,
        )
        if path:
            self.json_var.set(path)

    def _validate_float(self, value: str, name: str, min_value: float | None = None) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字") from exc
        if min_value is not None and parsed < min_value:
            raise ValueError(f"{name} 必须大于等于 {min_value}")
        return parsed

    def _validate_int(self, value: str, name: str, min_value: int | None = None) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} 必须是整数") from exc
        if min_value is not None and parsed < min_value:
            raise ValueError(f"{name} 必须大于等于 {min_value}")
        return parsed

    def _normalize_path(self, raw: str, suffix: str) -> pathlib.Path:
        path = pathlib.Path(raw).expanduser()
        if not path.name or str(path).endswith(("\\", "/")) or path.exists() and path.is_dir():
            path = path / f"calibration{suffix}"
        elif path.suffix.lower() != suffix:
            path = path.with_suffix(suffix)
        return path.resolve()

    def _start(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo(APP_TITLE, "正在标定，请等待当前任务结束。")
            return

        try:
            csv_path = self._normalize_path(self.csv_var.get(), ".csv")
            json_path = self._normalize_path(self.json_var.get(), ".json")
            count = self._validate_int(self.count_var.get(), "标定次数", 1)
            duration = self._validate_float(self.duration_var.get(), "每次时长", 1.0)
            beam = self._validate_float(self.beam_var.get(), "波束角度")
            distance = self._validate_float(self.distance_var.get(), "目标距离", 0.0)
            rms = self._validate_float(self.rms_var.get(), "RMS 门限", 0.0)
            attack = self._validate_float(self.attack_var.get(), "Attack 时间", 0.0)
            hold = self._validate_float(self.hold_var.get(), "Hold 时间", 0.0)
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_var.set(str(csv_path))
        self.json_var.set(str(json_path))

        self.stop_event = threading.Event()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("标定中")
        self._log("\n--- 开始标定 ---\n")

        self.worker = threading.Thread(
            target=self._worker,
            args=(csv_path, json_path, count, duration, beam, distance, rms, attack, hold, self.device_var.get().strip() or None),
            daemon=True,
        )
        self.worker.start()

    def _stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("正在停止...")

    def _on_keypress(self, _event) -> None:
        if self.stop_event is not None and self.start_button["state"] == "disabled":
            self.stop_event.set()

    def _worker(
        self,
        csv_path: pathlib.Path,
        json_path: pathlib.Path,
        count: int,
        duration: float,
        beam: float,
        distance: float,
        rms: float,
        attack: float,
        hold: float,
        device_hint: str | None,
    ) -> None:
        results: list[RunResult] = []
        try:
            for i in range(1, count + 1):
                if self.stop_event is not None and self.stop_event.is_set():
                    break

                wav_name = f"{self.prefix_var.get().strip() or 'calibration'}_{i:02d}.wav"
                wav_path = csv_path.parent / wav_name

                writer = TextQueueWriter(self.log_queue)
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    stats = recorder.record(
                        output_path=str(wav_path),
                        duration_sec=duration,
                        rms_threshold=rms,
                        device_hint=device_hint,
                        beam_deg=beam,
                        attack_ms=attack,
                        hold_ms=hold,
                        enable_spatial=False,
                        stop_event=self.stop_event,
                        trigger_on_voice=False,
                    )

                if not wav_path.exists():
                    raise RuntimeError(f"未生成 WAV 文件：{wav_path.name}")

                wav_duration_sec = self._wav_duration(wav_path)
                result = RunResult(
                    index=i,
                    filename=wav_path.name,
                    wav_path=wav_path,
                    peak_rms=stats.peak_rms,
                    wav_duration_sec=wav_duration_sec,
                    beam_deg=beam,
                    note=self.note_var.get().strip() or f"距离约 {distance:.1f}m",
                )
                results.append(result)

                if self.playback_var.get():
                    self.log_queue.put(f"播放 {wav_path.name}：{self._playback(wav_path)}\n")

                self.log_queue.put(
                    f"[{i}/{count}] {wav_path.name} 完成，Peak RMS={stats.peak_rms:.4f}，"
                    f"时长={wav_duration_sec:.2f}s\n"
                )

            if not results:
                raise RuntimeError("标定未产生任何有效结果。")

            self._write_csv(csv_path, results)
            ref_energy = statistics.median(r.peak_rms for r in results)
            self._write_json(json_path, results, beam, distance, ref_energy, duration, rms, attack, hold, device_hint)

            self.log_queue.put("\n--- 标定完成 ---\n")
            self.log_queue.put(f"CSV 文件：{csv_path}\n")
            self.log_queue.put(f"JSON 文件：{json_path}\n")
            self.log_queue.put(f"参考能量 ref_energy：{ref_energy:.6f}\n")
            self.log_queue.put("各次 Peak RMS：\n")
            for row in results:
                self.log_queue.put(f"  {row.index:02d}: {row.peak_rms:.6f} ({row.filename})\n")

            self.log_queue.put("__STATUS__DONE")
        except Exception as exc:
            self.log_queue.put(f"\n错误：{exc}\n")
            self.log_queue.put("__STATUS__ERROR")

    def _write_csv(self, csv_path: pathlib.Path, rows: list[RunResult]) -> None:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["文件名", "Peak RMS", "测试序号", "波束角度", "备注"])
            for row in rows:
                writer.writerow(
                    [
                        row.filename,
                        f"{row.peak_rms:.4f}",
                        row.index,
                        f"{row.beam_deg:.0f}°",
                        row.note,
                    ]
                )

    def _write_json(
        self,
        json_path: pathlib.Path,
        rows: list[RunResult],
        beam: float,
        distance: float,
        ref_energy: float,
        duration: float,
        rms: float,
        attack: float,
        hold: float,
        device_hint: str | None,
    ) -> None:
        import json

        data = {
            "target_angle_deg": beam,
            "target_distance_m": distance,
            "ref_energy": round(ref_energy, 6),
            "ref_rms": round(ref_energy, 6),
            "runs": len(rows),
            "duration_sec": duration,
            "rms_threshold": rms,
            "attack_ms": attack,
            "hold_ms": hold,
            "device_hint": device_hint,
            "peak_rms_values": [round(row.peak_rms, 6) for row in rows],
            "wav_files": [row.filename for row in rows],
            "calibrated_at": datetime.now().isoformat(),
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _wav_duration(self, path: pathlib.Path) -> float:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()

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
    app = CalibrationApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
