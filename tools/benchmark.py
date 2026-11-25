#!/usr/bin/env python
"""
Mede a velocidade REAL de cada modelo/dispositivo na SUA máquina (use no MacBook M1).
Compara, por exemplo, YOLOv8n x YOLO11n x YOLO26n em MPS e CPU.

    python tools/benchmark.py --models yolov8n.pt yolo11n.pt yolo26n.pt --devices mps cpu
    python tools/benchmark.py --models models/reciclagem_best.pt --imgsz 640 416 --frames 80
"""
import argparse
import os
import statistics
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt"])
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--video", default=None, help="vídeo real (senão usa ruído sintético: mede só velocidade, não precisão)")
    a = ap.parse_args()
    from ultralytics import YOLO
    import cv2

    frames = []
    if a.video:
        cap = cv2.VideoCapture(a.video)
        while len(frames) < a.frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    if not frames:
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(4)]

    print(f"{'modelo':<26}{'device':<8}{'imgsz':>6}{'mediana ms':>12}{'p95 ms':>9}{'FPS':>7}")
    for mp in a.models:
        try:
            model = YOLO(mp)
        except Exception as e:
            print(f"{mp:<26} não carregou: {e}")
            continue
        for dev in a.devices:
            for sz in a.imgsz:
                try:
                    for _ in range(5):   # aquecimento
                        model.predict(frames[0], imgsz=sz, device=dev, verbose=False)
                    ts = []
                    for i in range(a.frames):
                        t0 = time.perf_counter()
                        model.predict(frames[i % len(frames)], imgsz=sz, device=dev, verbose=False)
                        ts.append((time.perf_counter() - t0) * 1000)
                    med = statistics.median(ts)
                    p95 = sorted(ts)[int(0.95 * len(ts)) - 1]
                    print(f"{mp:<26}{dev:<8}{sz:>6}{med:>12.1f}{p95:>9.1f}{1000 / med:>7.1f}")
                except Exception as e:
                    print(f"{mp:<26}{dev:<8}{sz:>6}  falhou: {str(e)[:60]}")


if __name__ == "__main__":
    main()
