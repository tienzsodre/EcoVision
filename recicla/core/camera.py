"""
Captura de vídeo em thread própria.

* Guarda só o ÚLTIMO quadro (memória constante; se a IA for mais lenta que a câmera,
  quadros antigos são descartados em vez de formar fila e aumentar a latência).
* Aceita índice de câmera OU arquivo de vídeo (modo demonstração, com repetição).
* Reconecta sozinha se a câmera parar de entregar quadros.
* A ABERTURA (`open`) deve ser chamada na thread principal: no macOS é lá que o
  sistema pede/consulta a permissão de câmera (igual ao seu main.py original).
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Callable

import cv2
import numpy as np

from ..config import CameraCfg
from ..paths import resolve

StatusCb = Callable[[str, str], None]  # (nivel: info|warn|error, mensagem)

PERMISSION_HINT = ("No macOS confira: Ajustes do Sistema > Privacidade e Segurança > Câmera "
                   "(permita o Terminal/PyCharm/VS Code) e teste outro índice com tools/list_cameras.py.")


class CameraSource:
    def __init__(self, cfg: CameraCfg, status_cb: StatusCb | None = None):
        self.cfg = cfg
        self._status = status_cb or (lambda lvl, msg: None)
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self.capture_fps = 0.0
        self.failed = False
        self.actual = (0, 0, 0.0)  # largura, altura, fps reais negociados com a câmera

    @property
    def is_file(self) -> bool:
        return isinstance(self.cfg.source, str)

    # ---------------------------------------------------------------- abertura
    def open(self) -> bool:
        self._release()
        src = self.cfg.source
        if self.is_file:
            cap = cv2.VideoCapture(str(resolve(src)))
        elif sys.platform == "darwin":
            cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            what = f"o vídeo '{src}'" if self.is_file else f"a câmera de índice {src}"
            hint = "" if self.is_file else " " + PERMISSION_HINT
            self._status("error", f"Não foi possível abrir {what}.{hint}")
            return False
        if not self.is_file:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.height))
            cap.set(cv2.CAP_PROP_FPS, int(self.cfg.fps))
        self._cap = cap
        self.actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       float(cap.get(cv2.CAP_PROP_FPS)))
        self.failed = False
        return True

    def start(self) -> None:
        if self._cap is None:
            raise RuntimeError("chame open() antes de start()")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._release()
        with self._cond:
            self._frame = None

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ---------------------------------------------------------------- consumo
    def wait_frame(self, last_id: int, timeout: float = 0.5) -> tuple[np.ndarray | None, int]:
        """Bloqueia até existir um quadro mais novo que `last_id` (ou estourar o timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id or self._stop.is_set(), timeout)
            if self._frame is None or self._frame_id == last_id:
                return None, last_id
            return self._frame, self._frame_id

    # ---------------------------------------------------------------- thread
    def _run(self) -> None:
        attempts = 0
        while not self._stop.is_set():
            if self._cap is None:
                if attempts >= self.cfg.reconnect_attempts:
                    self.failed = True
                    self._status("error", "Câmera indisponível após várias tentativas. " + PERMISSION_HINT)
                    return
                attempts += 1
                self._status("warn", f"Reconectando à câmera (tentativa {attempts}/{self.cfg.reconnect_attempts})…")
                if self._stop.wait(1.0):
                    return
                if not self.open():
                    continue
            if self._read_loop():
                return  # parada pedida
            self._release()  # perdeu o sinal: tenta reabrir
            self._status("warn", "Sinal da câmera perdido.")
        # fim

    def _read_loop(self) -> bool:
        """Retorna True se terminou por `stop`, False se a câmera falhou."""
        cap = self._cap
        fails = 0
        file_fps = (self.actual[2] if self.actual[2] > 1 else 30.0) if self.is_file else 0.0
        file_dt = 1.0 / file_fps if file_fps else 0.0
        last = time.perf_counter()
        wraps_without_frame = 0
        while not self._stop.is_set() and cap is not None and cap.isOpened():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                if self.is_file:
                    wraps_without_frame += 1
                    if wraps_without_frame > 3:
                        return False
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # repete o vídeo
                    continue
                fails += 1
                if fails >= 30:
                    return False
                time.sleep(0.02)
                continue
            fails = 0
            wraps_without_frame = 0
            if self.cfg.mirror:
                frame = cv2.flip(frame, 1)
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            self.capture_fps = inst if self.capture_fps == 0 else 0.9 * self.capture_fps + 0.1 * inst
            last = now
            if file_dt:  # vídeo: respeita a velocidade original
                time.sleep(max(0.0, file_dt - (time.perf_counter() - t0)))
        return self._stop.is_set()
