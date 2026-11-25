"""Caminhos centrais do projeto (tudo relativo à raiz, funciona de qualquer diretório)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models"
SESSIONS_DIR = ROOT / "sessions"


def resolve(p: str | Path) -> Path:
    """Converte um caminho relativo do config em absoluto (relativo à raiz do projeto)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p
