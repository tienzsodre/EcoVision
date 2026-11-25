"""
YOLO + ByteTrack (embutido no Ultralytics), 100% local.

* Escolhe o dispositivo (MPS no Apple Silicon, senão CUDA, senão CPU).
* Converte as classes do modelo em materiais do `materials.yaml`; classes que não
  são materiais (ex.: "person" no modelo COCO) são ignoradas.
* Sem pesos treinados em recicláveis, cai no modelo COCO em modo DEMO (e avisa).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import MaterialsCfg, ModelCfg, TrackingCfg
from ..paths import resolve
from .types import RawTrack


@dataclass
class ModelInfo:
    weights: str
    device: str
    is_demo: bool
    class_names: list[str]
    mapped_classes: list[str]
    warnings: list[str]


def pick_device(requested: str) -> str:
    import torch
    req = (requested or "auto").lower()
    if req != "auto":
        return req
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class YoloTracker:
    def __init__(self, model_cfg: ModelCfg, tracking_cfg: TrackingCfg, materials: MaterialsCfg):
        self.cfg = model_cfg
        self.tracking_cfg = tracking_cfg
        self.materials = materials
        self.model = None
        self.device = "cpu"
        self.info: ModelInfo | None = None
        self._class_to_material: dict[int, str | None] = {}
        # ajustáveis em tempo real pela interface:
        self.conf = model_cfg.conf
        self.iou = model_cfg.iou

    # ------------------------------------------------------------------ carga
    def load(self, warmup_shape: tuple[int, int] = (720, 1280)) -> ModelInfo:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from ultralytics import YOLO  # import tardio: só custa quando o modelo é realmente usado

        custom = resolve(self.cfg.path)
        is_demo = not custom.exists()
        weights = resolve(self.cfg.fallback) if is_demo else custom
        weights.parent.mkdir(parents=True, exist_ok=True)
        # Se o fallback não existir, o Ultralytics baixa "yolo11n.pt" sozinho para esse caminho.
        self.model = YOLO(str(weights))
        self.device = pick_device(self.cfg.device)

        names: dict[int, str] = dict(self.model.names)
        warnings: list[str] = []
        self._class_to_material = {}
        for cid, name in names.items():
            m = self.materials.lookup_class(name)
            if m is None and is_demo:
                m = self.materials.lookup_demo(name)
            self._class_to_material[cid] = m.id if m else None
        mapped = [n for c, n in names.items() if self._class_to_material.get(c)]
        if not mapped:
            warnings.append("Nenhuma classe do modelo corresponde aos materiais de config/materials.yaml. "
                            f"Classes do modelo: {list(names.values())[:12]}…")
        if is_demo:
            warnings.append("MODELO DEMO (COCO): não foi treinado em recicláveis; classes mapeadas só "
                            "demonstram o pipeline. Treine com training/train.py.")

        self._warmup(warmup_shape)
        self.info = ModelInfo(str(weights), self.device, is_demo, list(names.values()), mapped, warnings)
        return self.info

    def _warmup(self, shape: tuple[int, int]) -> None:
        blank = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        try:
            self._predict(blank)
        except Exception:
            if self.device != "cpu":      # MPS/CUDA indisponível de fato -> CPU
                self.device = "cpu"
                self._predict(blank)
            else:
                raise
        self.reset_tracker()

    # ------------------------------------------------------------------ uso
    def _predict(self, frame_bgr: np.ndarray):
        kwargs = {}
        if self.cfg.half:  # só repassa se ligado: em versões novas do Ultralytics `half` está obsoleto
            kwargs["half"] = True
        return self.model.track(
            frame_bgr, persist=True, tracker=str(resolve(self.tracking_cfg.tracker_cfg)),
            conf=self.conf, iou=self.iou, imgsz=self.cfg.imgsz, device=self.device,
            max_det=self.cfg.max_det, verbose=False, **kwargs,
        )

    def infer(self, frame_bgr: np.ndarray) -> list[RawTrack]:
        try:
            results = self._predict(frame_bgr)
        except Exception:
            if self.device == "cpu":
                raise
            # Falha no acelerador (ex.: operador sem suporte no MPS): cai para CPU uma única vez.
            self.device = "cpu"
            results = self._predict(frame_bgr)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(clss)
        out: list[RawTrack] = []
        for (x1, y1, x2, y2), c, k, tid in zip(xyxy, confs, clss, ids):
            mat = self._class_to_material.get(int(k))
            if mat is None:
                continue
            out.append(RawTrack(None if tid is None else int(tid), mat, float(c),
                                (float(x1), float(y1), float(x2), float(y2))))
        return out

    def reset_tracker(self) -> None:
        """Zera os IDs do ByteTrack (câmera reiniciada)."""
        pred = getattr(self.model, "predictor", None)
        for t in getattr(pred, "trackers", None) or []:
            t.reset()
