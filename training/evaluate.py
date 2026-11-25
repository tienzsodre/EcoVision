#!/usr/bin/env python
"""
Avalia o modelo (precisão, recall, mAP) por classe no conjunto de TESTE (ou val).
Gera matriz de confusão em runs/eval/<name>/ e um relatório .json.

    python training/evaluate.py
    python training/evaluate.py --weights runs/detect/reciclagem/weights/best.pt --split val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/reciclagem/weights/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets/reciclagem/data.yaml"))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--name", default="reciclagem")
    args = ap.parse_args()
    for f in (args.weights, args.data):
        if not Path(f).exists():
            sys.exit(f"Arquivo não encontrado: {f}")

    from ultralytics import YOLO
    sys.path.insert(0, str(ROOT / "training"))
    from train import pick_device

    model = YOLO(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=pick_device(args.device),
                  project=str(ROOT / "runs/eval"), name=args.name, exist_ok=True, plots=True, verbose=False)
    b = m.box
    per_class = {}
    for i, ci in enumerate(b.ap_class_index):
        per_class[m.names[int(ci)]] = {"precisao": float(b.p[i]), "recall": float(b.r[i]),
                                       "mAP50": float(b.ap50[i]), "mAP50-95": float(b.ap[i])}
    n_inst = getattr(m, "nt_per_class", None)
    print(f"\n=== Avaliação no conjunto '{args.split}' ({args.weights}) ===")
    print(f"Geral: precisão {b.mp:.3f} | recall {b.mr:.3f} | mAP50 {b.map50:.3f} | mAP50-95 {b.map:.3f}")
    print(f"{'classe':<22}{'inst.':>7}{'prec.':>8}{'recall':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for ci, name in m.names.items():
        if name in per_class:
            r = per_class[name]
            n = int(n_inst[ci]) if n_inst is not None and ci < len(n_inst) else -1
            print(f"{name:<22}{n:>7}{r['precisao']:>8.3f}{r['recall']:>8.3f}{r['mAP50']:>8.3f}{r['mAP50-95']:>10.3f}")
        else:
            print(f"{name:<22}{'0':>7}   (sem exemplos no conjunto '{args.split}': NÃO foi avaliada)")
    out = ROOT / "runs/eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"relatorio_{args.split}.json").write_text(json.dumps(
        {"weights": args.weights, "split": args.split,
         "geral": {"precisao": float(b.mp), "recall": float(b.mr), "mAP50": float(b.map50), "mAP50-95": float(b.map)},
         "por_classe": per_class}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRelatório e matriz de confusão em {out}")
    print("Dica: olhe a matriz de confusão — lata de alumínio x lata de aço e PET x plástica costumam se confundir.")


if __name__ == "__main__":
    main()
