#!/usr/bin/env python
"""
Descobre quais índices de câmera funcionam (evolução do teste_camera.py, que só testava o 0).
No Mac, 0/1/2 podem ser: câmera do MacBook, iPhone (Continuidade) ou webcam USB.

    python tools/list_cameras.py          # lista índices 0..5 com resolução
    python tools/list_cameras.py --show   # abre uma janela de prévia de cada câmera (q = próxima)
"""
import argparse
import sys

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    found = []
    for i in range(a.max):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            h, w = frame.shape[:2]
            print(f"índice {i}: OK  {w}x{h}")
            found.append(i)
            while a.show:
                ok, frame = cap.read()
                if not ok:
                    break
                cv2.imshow(f"camera {i} (q = proxima)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
        else:
            print(f"índice {i}: sem imagem")
        cap.release()
    print("\nUse o índice desejado em Configurações > Câmera (ou config/settings.yaml: camera.source)." if found
          else "\nNenhuma câmera entregou imagem. No macOS, libere a permissão em Ajustes > Privacidade > Câmera.")


if __name__ == "__main__":
    main()
