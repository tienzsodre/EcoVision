"""
Regra de contagem: cada objeto físico é contado UMA vez.

O rastreador (ByteTrack) dá um ID a cada objeto. Aqui decidimos QUANDO contar:

1. Confirmação: o ID só é contado depois de `min_hits` quadros (evita falsos
   positivos de 1-2 quadros).
2. Voto de classe: a categoria final é a de maior confiança acumulada ao longo da
   trilha (evita contar "PET" e depois "plástica" para o mesmo objeto piscando).
3. Reidentificação: se o rastreador perde um objeto e devolve outro ID (oclusão,
   movimento rápido), e esse ID novo aparece onde um objeto JÁ CONTADO sumiu há
   poucos segundos, com a mesma categoria e caixa sobreposta (IoU), ele é tratado
   como o mesmo objeto e NÃO é contado de novo.

Limite conhecido (inerente a qualquer câmera única): um objeto que sai de cena e
volta depois da janela de reidentificação é contado novamente; e um objeto
diferente colocado exatamente no lugar de outro dentro da janela pode ser
confundido com ele (por isso `reid_window_s` é configurável, 0 desliga).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .types import Box, CountEvent, Detection, RawTrack


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


@dataclass
class _TrackState:
    id: int
    last_seen: float
    hits: int = 0
    votes: dict[str, float] = field(default_factory=dict)
    conf_sum: float = 0.0
    last_box: Box = (0.0, 0.0, 0.0, 0.0)
    counted: bool = False

    @property
    def material(self) -> str:
        return max(self.votes, key=self.votes.get)

    @property
    def conf_mean(self) -> float:
        return self.conf_sum / max(self.hits, 1)


class CountingManager:
    def __init__(self, min_hits: int = 5, reid_window_s: float = 3.0, reid_iou: float = 0.4):
        self._lock = threading.Lock()
        self._states: dict[int, _TrackState] = {}
        self.min_hits = min_hits
        self.reid_window_s = reid_window_s
        self.reid_iou = reid_iou

    def configure(self, min_hits: int, reid_window_s: float, reid_iou: float) -> None:
        with self._lock:
            self.min_hits = max(1, int(min_hits))
            self.reid_window_s = max(0.0, float(reid_window_s))
            self.reid_iou = float(reid_iou)

    def reset(self) -> None:
        """Esquece todas as trilhas (nova sessão, câmera reiniciada, tracker resetado)."""
        with self._lock:
            self._states.clear()

    def update(self, tracks: list[RawTrack], now: float | None = None,
               wall: float | None = None, *,
               class_override: str | None = None) -> tuple[list[Detection], list[CountEvent]]:
        now = time.monotonic() if now is None else now
        wall = time.time() if wall is None else wall
        detections: list[Detection] = []
        events: list[CountEvent] = []
        with self._lock:
            current_ids = {t.track_id for t in tracks if t.track_id is not None}
            for t in tracks:
                if t.track_id is None:
                    detections.append(Detection(None, t.material, t.conf, t.box, "untracked", t.material))
                    continue
                st = self._states.get(t.track_id)
                if st is None:
                    st = self._states[t.track_id] = _TrackState(t.track_id, now)
                st.last_seen = now
                st.hits += 1
                st.votes[t.material] = st.votes.get(t.material, 0.0) + t.conf
                st.conf_sum += t.conf
                st.last_box = t.box

                if not st.counted and st.hits >= self.min_hits:
                    st.counted = True
                    if not self._merge_with_lost(st, current_ids, now):
                        if class_override is None:
                            events.append(CountEvent(st.id, st.material, st.conf_mean, wall))
                        else:
                            events.append(CountEvent(st.id, class_override, None, wall, st.material))
                state = "counted" if st.counted else "candidate"
                detections.append(Detection(st.id, st.material, t.conf, t.box, state, t.material))
            self._cleanup(now)
        return detections, events

    # -- internos (chamados com o lock já adquirido) ------------------------------
    def _merge_with_lost(self, new: _TrackState, current_ids: set[int], now: float) -> bool:
        """True se `new` for continuação de um objeto já contado que sumiu há pouco."""
        if self.reid_window_s <= 0:
            return False
        best, best_iou = None, self.reid_iou
        for old in self._states.values():
            if (old.id == new.id or not old.counted or old.id in current_ids
                    or now - old.last_seen > self.reid_window_s
                    or old.material != new.material):
                continue
            v = iou(old.last_box, new.last_box)
            if v >= best_iou:
                best, best_iou = old, v
        if best is None:
            return False
        del self._states[best.id]  # cada objeto antigo só "herda" uma vez
        return True

    def _cleanup(self, now: float) -> None:
        stale = self.reid_window_s + 2.0
        for tid in [k for k, s in self._states.items() if now - s.last_seen > stale]:
            del self._states[tid]
