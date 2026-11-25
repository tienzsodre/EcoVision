"""Sessão de triagem: contagens, histórico, estatísticas e exportação CSV (thread-safe)."""
from __future__ import annotations

import csv
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import MaterialsCfg
from ..paths import SESSIONS_DIR
from .estimates import group_estimate, item_estimate
from .types import CountEvent


@dataclass
class CategoryStat:
    material: str
    count: int
    weight_kg: float
    value: float


@dataclass
class Snapshot:
    total: int
    by_material: dict[str, CategoryStat]
    total_weight_kg: float
    total_value: float
    elapsed_s: float
    rate_last_min: int       # itens contados nos últimos 60 s
    rate_avg_per_min: float  # média da sessão
    started_at: float
    last_seq: int


class Session:
    def __init__(self, materials: MaterialsCfg, max_events: int = 20000):
        self._lock = threading.RLock()
        self.materials = materials
        self._events: deque[tuple[int, CountEvent]] = deque(maxlen=max_events)
        self._times: deque[float] = deque()
        self._counts: Counter[str] = Counter()
        self._seq = 0
        self.started_at = time.time()

    # -- escrita (thread de IA) ----------------------------------------------------
    def add_event(self, ev: CountEvent) -> None:
        with self._lock:
            self._seq += 1
            self._events.append((self._seq, ev))
            self._counts[ev.material] += 1
            self._times.append(ev.timestamp)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._times.clear()
            self._counts.clear()
            self._seq = 0
            self.started_at = time.time()

    # -- leitura (interface) -------------------------------------------------------
    def events_since(self, seq: int) -> list[tuple[int, CountEvent]]:
        with self._lock:
            return [(s, e) for s, e in self._events if s > seq]

    def snapshot(self) -> Snapshot:
        now = time.time()
        with self._lock:
            counts = dict(self._counts)
            while self._times and now - self._times[0] > 60:
                self._times.popleft()
            last_min = len(self._times)
            seq, started = self._seq, self.started_at
        by, tw, tv = {}, 0.0, 0.0
        for m in self.materials:
            c = counts.get(m.id, 0)
            w, v = group_estimate(m, c)
            by[m.id] = CategoryStat(m.id, c, w, v)
            tw += w
            tv += v
        total = sum(counts.values())
        elapsed = max(now - started, 1e-6)
        return Snapshot(total, by, tw, tv, elapsed, last_min, total / (elapsed / 60.0), started, seq)

    def has_data(self) -> bool:
        with self._lock:
            return self._seq > 0

    # -- exportação ----------------------------------------------------------------
    def export_csv(self, directory: Path = SESSIONS_DIR) -> tuple[Path, Path]:
        """Grava <sessao>_eventos.csv e <sessao>_resumo.csv. Estimativas usam a config ATUAL."""
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            events = list(self._events)
            snap = self.snapshot()
        stamp = datetime.fromtimestamp(snap.started_at).strftime("%Y%m%d_%H%M%S")
        ev_path = directory / f"sessao_{stamp}_eventos.csv"
        sum_path = directory / f"sessao_{stamp}_resumo.csv"
        with open(ev_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["horario", "id_rastreamento", "categoria_id", "categoria", "confianca",
                        "peso_estimado_g (ESTIMATIVA)", "valor_estimado (ESTIMATIVA)"])
            for _, e in events:
                m = self.materials.get(e.material)
                kg, val = item_estimate(m) if m else (0.0, 0.0)
                w.writerow([datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                            e.track_id, e.material, m.label if m else e.material,
                            f"{e.conf:.2f}" if e.conf is not None else "",
                            f"{kg * 1000:.1f}", f"{val:.2f}"])
        with open(sum_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["categoria", "quantidade", "peso_medio_g", f"preco_por_kg ({self.materials.currency})",
                        "peso_estimado_kg (ESTIMATIVA)", "valor_estimado (ESTIMATIVA)"])
            for m in self.materials:
                s = snap.by_material[m.id]
                w.writerow([m.label, s.count, m.avg_weight_g, m.price_per_kg,
                            f"{s.weight_kg:.3f}", f"{s.value:.2f}"])
            w.writerow(["TOTAL", snap.total, "", "", f"{snap.total_weight_kg:.3f}", f"{snap.total_value:.2f}"])
            w.writerow([])
            w.writerow(["Pesos e valores são ESTIMATIVAS (contagem x peso médio x preço/kg), não medições."])
        return ev_path, sum_path

    def append_history_line(self, directory: Path = SESSIONS_DIR) -> Path | None:
        """Acrescenta uma linha-resumo em historico_sessoes.csv (histórico de sessões)."""
        if not self.has_data():
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "historico_sessoes.csv"
        new = not path.exists()
        snap = self.snapshot()
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            if new:
                w.writerow(["inicio", "fim", "duracao_s", "total_itens",
                            "peso_estimado_kg", "valor_estimado"])
            fmt = "%Y-%m-%d %H:%M:%S"
            w.writerow([datetime.fromtimestamp(self.started_at).strftime(fmt),
                        datetime.now().strftime(fmt), int(snap.elapsed_s), snap.total,
                        f"{snap.total_weight_kg:.3f}", f"{snap.total_value:.2f}"])
        return path
