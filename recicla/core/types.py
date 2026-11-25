"""Estruturas de dados trocadas entre as threads (câmera -> IA -> interface)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels do quadro original)


@dataclass(slots=True)
class RawTrack:
    """Uma detecção já com ID de rastreamento, vinda do YOLO + ByteTrack."""
    track_id: int | None
    material: str
    conf: float
    box: Box


@dataclass(slots=True)
class Detection:
    """Detecção pronta para desenhar. state: untracked | candidate | counted."""
    track_id: int | None
    material: str
    conf: float | None
    box: Box
    state: str
    predicted_material: str | None = None


@dataclass(slots=True)
class CountEvent:
    """Um objeto novo que acabou de ser confirmado e contabilizado (uma única vez)."""
    track_id: int
    material: str
    conf: float | None
    timestamp: float  # epoch (time.time())
    predicted_material: str | None = None


@dataclass(slots=True)
class FrameResult:
    frame_id: int
    frame: np.ndarray            # BGR, SEM anotações (a interface desenha por cima)
    detections: list[Detection] = field(default_factory=list)
    capture_fps: float = 0.0
    infer_fps: float = 0.0
    infer_ms: float = 0.0
