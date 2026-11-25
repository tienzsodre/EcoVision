"""
Carregamento/gravação de configurações.

* settings.yaml / materials.yaml  -> padrões versionados (com comentários)
* settings.local.yaml / materials.local.yaml -> o que a tela de Configurações grava
  (sobrepõe os padrões; não perde os comentários dos arquivos originais).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR


# ----------------------------------------------------------------------------- helpers
def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _build(cls, data: dict | None):
    """Cria o dataclass usando apenas as chaves conhecidas (ignora lixo no YAML)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


def normalize_name(name: str) -> str:
    """'Clear plastic bottle' -> 'clear_plastic_bottle' (comparação tolerante de nomes)."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def parse_source(value: Any) -> int | str:
    """'1' -> 1 (índice de câmera); '/x/video.mp4' -> caminho de arquivo."""
    if isinstance(value, int):
        return value
    s = str(value).strip()
    return int(s) if s.lstrip("-").isdigit() else s


# ----------------------------------------------------------------------------- settings
@dataclass
class CameraCfg:
    source: int | str = 1
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = False
    reconnect_attempts: int = 5


@dataclass
class ModelCfg:
    path: str = "models/reciclagem_best.pt"
    fallback: str = "models/yolo11n.pt"
    device: str = "auto"
    imgsz: int = 640
    conf: float = 0.40
    iou: float = 0.50
    max_det: int = 50
    half: bool = False
    max_infer_fps: int = 30


@dataclass
class TrackingCfg:
    tracker_cfg: str = "config/bytetrack_recycle.yaml"
    min_hits: int = 5
    reid_window_s: float = 3.0
    reid_iou: float = 0.40


@dataclass
class UiCfg:
    history_rows: int = 200
    stats_refresh_ms: int = 250


@dataclass
class Settings:
    camera: CameraCfg = field(default_factory=CameraCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    ui: UiCfg = field(default_factory=UiCfg)

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> "Settings":
        merged: dict = {}
        for name in ("settings.yaml", "settings.local.yaml"):
            for section, values in _read_yaml(config_dir / name).items():
                merged.setdefault(section, {}).update(values or {})
        s = cls(
            camera=_build(CameraCfg, merged.get("camera")),
            model=_build(ModelCfg, merged.get("model")),
            tracking=_build(TrackingCfg, merged.get("tracking")),
            ui=_build(UiCfg, merged.get("ui")),
        )
        s.camera.source = parse_source(s.camera.source)
        return s

    def save(self, config_dir: Path = CONFIG_DIR) -> Path:
        path = config_dir / "settings.local.yaml"
        _write_yaml(path, asdict(self))
        return path


# ----------------------------------------------------------------------------- materials
@dataclass
class Material:
    id: str
    label: str
    color: str = "#ffffff"
    avg_weight_g: float = 0.0
    price_per_kg: float = 0.0
    model_classes: list[str] = field(default_factory=list)
    demo_coco: list[str] = field(default_factory=list)


class MaterialsCfg:
    """Catálogo de materiais (ordem do YAML = ordem na interface)."""

    def __init__(self, currency: str, materials: dict[str, Material]):
        self.currency = currency
        self.materials = materials
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_class: dict[str, Material] = {}
        self._by_demo: dict[str, Material] = {}
        for m in self.materials.values():
            for alias in [m.id, *m.model_classes]:
                self._by_class[normalize_name(alias)] = m
            for alias in m.demo_coco:
                self._by_demo[normalize_name(alias)] = m

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> "MaterialsCfg":
        base = _read_yaml(config_dir / "materials.yaml")
        local = _read_yaml(config_dir / "materials.local.yaml")
        currency = local.get("currency", base.get("currency", "R$"))
        out: dict[str, Material] = {}
        for mid, raw in (base.get("materials") or {}).items():
            raw = dict(raw or {})
            raw.update((local.get("materials") or {}).get(mid, {}))  # só peso/preço
            raw.setdefault("label", mid)
            out[mid] = _build(Material, {**raw, "id": mid})
        return cls(currency, out)

    def save(self, config_dir: Path = CONFIG_DIR) -> Path:
        """Grava só o que a interface edita (peso médio e preço/kg)."""
        path = config_dir / "materials.local.yaml"
        _write_yaml(path, {
            "currency": self.currency,
            "materials": {
                m.id: {"avg_weight_g": float(m.avg_weight_g), "price_per_kg": float(m.price_per_kg)}
                for m in self.materials.values()
            },
        })
        return path

    def __iter__(self):
        return iter(self.materials.values())

    def get(self, mid: str) -> Material | None:
        return self.materials.get(mid)

    def lookup_class(self, model_class_name: str) -> Material | None:
        return self._by_class.get(normalize_name(model_class_name))

    def lookup_demo(self, coco_class_name: str) -> Material | None:
        return self._by_demo.get(normalize_name(coco_class_name))
