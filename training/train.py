#!/usr/bin/env python
"""
Treina o YOLO localmente (MPS no Apple Silicon, CUDA se houver, senão CPU).

    python training/train.py                      # YOLO11n, 100 épocas, 640 px
    python training/train.py --base yolo26n.pt    # compare no seu Mac com tools/benchmark.py
    python training/train.py --epochs 1 --fraction 0.05   # teste rápido só p/ ver se tudo roda

O resultado fica em runs/detect/<name>/weights/best.pt. Depois:
    python training/evaluate.py     # métricas no conjunto de teste
    python training/export_model.py # copia p/ models/reciclagem_best.pt (o app passa a usar)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parent.parent


def pick_device(req: str) -> str:
    import torch
    if req != "auto":
        return req
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "datasets/reciclagem/data.yaml"))
    ap.add_argument("--base", default="yolo11n.pt", help="pesos iniciais (transfer learning): yolo11n.pt, yolov8n.pt, yolo26n.pt…")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="reduza p/ 8 se faltar memória (MacBook Air 8 GB)")
    ap.add_argument("--device", default="auto", help="auto | mps | cpu | 0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30, help="para cedo se a validação não melhorar")
    ap.add_argument("--fraction", type=float, default=1.0, help="usa só uma fração do treino (testes rápidos)")
    ap.add_argument("--name", default="reciclagem")
    ap.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    ap.add_argument("--cache", default="False", help="False | ram | disk (ram acelera, mas gasta memória)")
    ap.add_argument("--resume", action="store_true", help="retoma o último treino interrompido deste --name")
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        sys.exit(f"{data} não existe. Rode antes:  python training/prepare_dataset.py")

    from ultralytics import YOLO

    device = pick_device(args.device)
    print(f"Dispositivo: {device} | base: {args.base} | dados: {data}")
    if args.resume:
        last = Path(args.project) / args.name / "weights" / "last.pt"
        model = YOLO(str(last))
        model.train(resume=True)
        return

    model = YOLO(args.base)
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        workers=args.workers, patience=args.patience, fraction=args.fraction,
        project=args.project, name=args.name, exist_ok=True,
        cache={"False": False, "ram": "ram", "disk": "disk"}.get(args.cache, False),
        # Augmentações escolhidas p/ objetos soltos em mesa/esteira (sem exagero):
        fliplr=0.5, flipud=0.2, degrees=15, scale=0.5, translate=0.1,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, mosaic=1.0, close_mosaic=10,
        seed=42, plots=True,
    )
    best = getattr(model.trainer, "best", None) or Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTreino terminado. Melhor modelo: {best}")
    print("Avalie com:  python training/evaluate.py")


if __name__ == "__main__":
    main()
