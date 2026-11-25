"""
Ponto de entrada:  python main.py

(O main.py original, de um único arquivo, foi preservado em legacy/main_original.py.)
"""
import os
import sys

# Operadores do PyTorch ainda sem suporte no MPS (Apple Silicon) caem para a CPU em vez de falhar.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from recicla.ui.app import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
