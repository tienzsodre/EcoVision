#!/usr/bin/env python
"""
Publica os pesos treinados para o app: copia best.pt -> models/reciclagem_best.pt
(e opcionalmente exporta CoreML p/ Apple Silicon), conferindo se as classes do modelo
batem com config/materials.yaml.

    python training/export_model.py
    python training/export_model.py --weights runs/detect/reciclagem/weights/best.pt --coreml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recicla.config import MaterialsCfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/reciclagem/weights/best.pt"))
    ap.add_argument("--dest", default=str(ROOT / "models/reciclagem_best.pt"))
    ap.add_argument("--coreml", action="store_true", help="também exporta .mlpackage (experimental; o app usa o .pt por padrão)")
    args = ap.parse_args()
    src = Path(args.weights)
    if not src.exists():
        sys.exit(f"Pesos não encontrados: {src}. Treine antes com training/train.py.")

    from ultralytics import YOLO
    model = YOLO(str(src))
    names = list(model.names.values())
    mats = MaterialsCfg.load()
    unknown = [n for n in names if mats.lookup_class(n) is None]
    if unknown:
        sys.exit(f"As classes {unknown} do modelo não existem em config/materials.yaml. "
                 "Ajuste o materials.yaml (ou o treino) para que os nomes coincidam.")
    missing = [m.id for m in mats if m.id not in [n.lower() for n in names]]
    if missing:
        print(f"Aviso: materiais sem classe correspondente no modelo (nunca serão detectados): {missing}")

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    (dest.with_suffix(".meta.json")).write_text(json.dumps({
        "origem": str(src), "exportado_em": datetime.now().isoformat(timespec="seconds"),
        "classes": names, "observacao": "Métricas reais: rode training/evaluate.py (não são gravadas aqui)."},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo copiado para {dest}. O app (python main.py) passa a usá-lo automaticamente.")
    if args.coreml:
        out = model.export(format="coreml", imgsz=640)
        print(f"CoreML exportado em {out} (experimental: aponte model.path p/ ele em Configurações para testar).")


if __name__ == "__main__":
    main()
