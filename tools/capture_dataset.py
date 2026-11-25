#!/usr/bin/env python
"""
Captura fotos com a SUA câmera para montar o dataset próprio (a melhor forma de melhorar o modelo,
pois é exatamente o ambiente/ângulo/iluminação de uso real).

    python tools/capture_dataset.py --source 1
    ESPAÇO = salva uma foto | a = liga/desliga captura automática (1 foto/s) | q = sai

Fotos vão para datasets/raw/meu_dataset/images/ e um classes.txt (ids de config/materials.yaml)
é gerado para você anotar com labelImg, CVAT ou Label Studio exportando em formato YOLO.
Depois ative a fonte `meu_dataset` em training/datasets.yaml.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recicla.config import MaterialsCfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="1")
    ap.add_argument("--out", default=str(ROOT / "datasets/raw/meu_dataset"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(m.id for m in MaterialsCfg.load()) + "\n", encoding="utf-8")
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit("Não consegui abrir a câmera/vídeo (veja tools/list_cameras.py).")
    n, auto, last = len(list((out / "images").glob("*.jpg"))), False, 0.0
    print("ESPAÇO salva | a = automático | q sai")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shown = frame.copy()
        cv2.putText(shown, f"fotos: {n}  auto: {'ON' if auto else 'off'}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("capturar dataset", shown)
        k = cv2.waitKey(1) & 0xFF
        save = k == ord(" ") or (auto and time.time() - last > 1.0)
        if k == ord("a"):
            auto = not auto
        if save:
            cv2.imwrite(str(out / "images" / f"cap_{int(time.time() * 1000)}.jpg"), frame)
            n, last = n + 1, time.time()
        if k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"{n} fotos em {out / 'images'}")


if __name__ == "__main__":
    main()
