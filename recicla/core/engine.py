"""
Motor do sistema: orquestra as threads e isola a interface do processamento pesado.

    [thread câmera] --último quadro--> [thread IA: YOLO + ByteTrack + contagem] --FrameResult--> [Tk]

* O Tkinter NUNCA é tocado fora da thread principal. As threads só publicam dados:
  `latest_result()` (último quadro processado) e a fila `events` (mensagens de status/erro).
  A interface consulta ambos periodicamente com `after()`.
* O modelo é carregado uma vez (na thread de IA, sem travar a janela) e reaproveitado
  entre "Encerrar" e "Iniciar".
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import replace
from typing import Callable

from ..config import MaterialsCfg, Settings
from .camera import CameraSource
from .counting import CountingManager
from .detector import YoloTracker
from .session import Session
from .types import FrameResult

LOGGER = logging.getLogger(__name__)
CLASS_OVERRIDES = ("garrafa", "lata")


class Engine:
    MAX_CONSECUTIVE_ERRORS = 10

    def __init__(self, settings: Settings, materials: MaterialsCfg, session: Session,
                 tracker_factory: Callable[[], YoloTracker] | None = None):
        self.settings = settings
        self.materials = materials
        self.session = session
        self._factory = tracker_factory or (lambda: YoloTracker(settings.model, settings.tracking, materials))
        self.counter = CountingManager(settings.tracking.min_hits, settings.tracking.reid_window_s,
                                       settings.tracking.reid_iou)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.camera: CameraSource | None = None
        self.tracker: YoloTracker | None = None
        self.state = "idle"   # idle | loading | running | paused
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._count_lock = threading.Lock()
        self._result: FrameResult | None = None
        self._automatic_result: FrameResult | None = None
        self._class_override: str | None = None
        self._logged_overrides: set[tuple[int | None, str]] = set()
        self._reset_counter = threading.Event()

    # ------------------------------------------------------------------ comandos (thread da UI)
    def start(self) -> bool:
        if self.state != "idle":
            return False
        self.camera = CameraSource(self.settings.camera, lambda lvl, msg: self._post("status", (lvl, msg)))
        if not self.camera.open():          # na thread principal (permissão de câmera no macOS)
            self.camera = None
            return False
        self._stop.clear()
        self._paused.clear()
        self.state = "loading"
        self._post("state", self.state)
        self.camera.start()
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)
        self._thread.start()
        return True

    def pause(self) -> None:
        if self.state == "running":
            self._paused.set()
            self.state = "paused"
            self._post("state", self.state)

    def resume(self) -> None:
        if self.state == "paused":
            self._paused.clear()
            self.state = "running"
            self._post("state", self.state)

    def stop(self) -> None:
        if self.state == "idle" and self._thread is None:
            return
        self._stop.set()
        self._paused.clear()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=8.0)
        self._thread = None
        if self.camera:
            self.camera.stop()
            self.camera = None
        with self._lock:
            self._automatic_result = self._result = None
            self._logged_overrides.clear()
        self.state = "idle"
        self._post("state", self.state)

    def restart_counting(self) -> None:
        """Nova sessão / zerar: o que está na tela volta a ser candidato a contagem."""
        self._reset_counter.set()

    def reset_session(self, archive: bool = False) -> None:
        """Encerra a contagem anterior entre atualizações do rastreador."""
        with self._count_lock:
            if archive and self.session.has_data():
                self.session.export_csv()
                self.session.append_history_line()
            self.session.reset()
            self.counter.reset()

    def invalidate_model(self) -> None:
        """Chame após mudar caminho/dispositivo/imgsz (só vale com o motor parado)."""
        self.tracker = None

    def apply_live_settings(self) -> None:
        t = self.settings.tracking
        self.counter.configure(t.min_hits, t.reid_window_s, t.reid_iou)
        if self.tracker:
            self.tracker.conf = self.settings.model.conf
            self.tracker.iou = self.settings.model.iou

    def set_conf(self, conf: float) -> None:
        self.settings.model.conf = float(conf)
        if self.tracker:
            self.tracker.conf = float(conf)

    def latest_result(self) -> FrameResult | None:
        with self._lock:
            return self._result

    @property
    def class_override(self) -> str | None:
        with self._lock:
            return self._class_override

    def set_class_override(self, material: str | None) -> None:
        if material is not None and (material not in CLASS_OVERRIDES or not self.materials.get(material)):
            raise ValueError("Material indisponível para classificação manual.")
        with self._count_lock, self._lock:
            if material == self._class_override:
                return
            LOGGER.info("Classificação: %s → %s", self._class_override or "automática", material or "automática")
            self._class_override = material
            self._logged_overrides.clear()
            self._update_display_result()

    def _update_display_result(self) -> None:
        """Aplica o modo atual ao quadro disponível, inclusive durante uma pausa."""
        result = self._automatic_result
        material = self._class_override
        if result is None or material is None:
            self._result = result
            return
        overrides = {(det.track_id, det.predicted_material or det.material) for det in result.detections}
        for track_id, predicted in overrides - self._logged_overrides:
            LOGGER.info("Classificação manual: id=%s previsão=%s classe efetiva=%s",
                        track_id if track_id is not None else "sem tracking", predicted, material)
        self._logged_overrides = overrides
        self._result = replace(result, detections=[
            replace(det, material=material, conf=None) for det in result.detections
        ])

    # ------------------------------------------------------------------ thread de IA
    def _post(self, kind: str, payload: object) -> None:
        self.events.put((kind, payload))

    def _run(self) -> None:
        try:
            if self.tracker is not None:
                self.tracker.reset_tracker()
            if not self.prepare_model():
                return
            with self._count_lock:
                self.counter.reset()
            self.state = "running" if not self._paused.is_set() else "paused"
            self._post("state", self.state)
            self._loop()
        except Exception as exc:  # qualquer falha inesperada vira mensagem, não trava a janela
            self._post("error", f"Erro no processamento: {exc}")
        finally:
            if not self._stop.is_set():          # parou sozinho (erro/câmera): UI deve voltar a "parado"
                self._post("stopped", None)

    def prepare_model(self) -> bool:
        """Carrega e aquece o modelo. Chamar sem inferência concorrente."""
        if self.tracker is None:
            self._post("status", ("info", "Carregando modelo de IA (primeira vez demora alguns segundos)…"))
            try:
                tr = self._factory()
                w, h, _ = (self.camera.actual if self.camera else
                           (self.settings.camera.width, self.settings.camera.height, 0))
                info = tr.load(warmup_shape=(h or 720, w or 1280))
            except Exception as exc:
                self._post("error", f"Não foi possível carregar o modelo: {exc}")
                return False
            self.tracker = tr
            self._post("model", info)
            for w_msg in info.warnings:
                self._post("status", ("warn", w_msg))
        self.tracker.conf = self.settings.model.conf
        self.tracker.iou = self.settings.model.iou
        return True

    def _loop(self) -> None:
        cam, tracker = self.camera, self.tracker
        min_dt = 1.0 / max(self.settings.model.max_infer_fps, 1)
        last_id, errors = -1, 0
        infer_fps, t_prev = 0.0, time.perf_counter()
        last_frame_t = time.monotonic()
        warned_nosignal = False
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            if self._reset_counter.is_set():
                self._reset_counter.clear()
                with self._count_lock:
                    self.counter.reset()
            frame, fid = cam.wait_frame(last_id, timeout=0.5)
            if frame is None:
                if cam.failed:
                    self._post("error", "A câmera parou de responder e não foi possível reconectar.")
                    return
                if time.monotonic() - last_frame_t > 5 and not warned_nosignal:
                    warned_nosignal = True
                    self._post("status", ("warn", "Sem imagem da câmera há 5 s…"))
                continue
            last_id, last_frame_t, warned_nosignal = fid, time.monotonic(), False

            t0 = time.perf_counter()
            try:
                raw = tracker.infer(frame)
                errors = 0
            except Exception as exc:
                errors += 1
                if errors == 1:
                    self._post("status", ("warn", f"Falha na inferência: {exc}"))
                if errors >= self.MAX_CONSECUTIVE_ERRORS:
                    self._post("error", f"Inferência falhou {errors} vezes seguidas: {exc}")
                    return
                continue
            infer_ms = (time.perf_counter() - t0) * 1000.0

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            infer_fps = inst if infer_fps == 0 else 0.9 * infer_fps + 0.1 * inst
            with self._count_lock:
                detections, new_events = self.counter.update(raw, class_override=self._class_override)
                for ev in new_events:
                    self.session.add_event(ev)
                with self._lock:
                    self._automatic_result = FrameResult(
                        fid, frame, detections, cam.capture_fps, infer_fps, infer_ms
                    )
                    self._update_display_result()

            spare = min_dt - (time.perf_counter() - t0)   # teto de FPS: poupa bateria/CPU
            if spare > 0:
                time.sleep(spare)
