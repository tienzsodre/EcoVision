#!/usr/bin/env python
"""
Converte/une datasets (COCO ou YOLO) no formato YOLO do projeto, com as classes de
config/materials.yaml, e divide em treino/validação/teste.

    python training/prepare_dataset.py                 # usa training/datasets.yaml
    python training/prepare_dataset.py --config outro.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recicla.config import MaterialsCfg, normalize_name  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Item:
    image: Path
    boxes: list[tuple[str, float, float, float, float]]   # (nome_origem, cx, cy, w, h) normalizados


def _p(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


# ----------------------------------------------------------------------------- leitores
def read_coco(src: dict) -> tuple[list[Item], set[str]]:
    ann_path = _p(src["annotations"])
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    imgs = {i["id"]: i for i in data["images"]}
    per_img: dict[int, list] = defaultdict(list)
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        im = imgs.get(a["image_id"])
        if im is None:
            continue
        x, y, w, h = a["bbox"]
        W, H = im["width"], im["height"]
        x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        per_img[a["image_id"]].append((cats[a["category_id"]], (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                                       (x2 - x1) / W, (y2 - y1) / H))
    root = _p(src["images_root"])
    items = []
    for iid, boxes in per_img.items():
        path = root / imgs[iid]["file_name"]
        if path.exists():
            items.append(Item(path, boxes))
    return items, set(cats.values())


def _names_for_yolo(src: dict, root: Path) -> list[str]:
    if src.get("names"):
        return list(src["names"])
    for y in list(root.glob("*.yaml")) + list(root.glob("*/*.yaml")):
        d = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        if "names" in d:
            n = d["names"]
            return [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
    for name in ("classes.txt", "obj.names", "labels.txt"):
        for f in list(root.glob(name)) + list(root.glob(f"*/{name}")):
            return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise SystemExit(f"[{src['name']}] não achei os nomes das classes (data.yaml/classes.txt). "
                     "Defina `names:` na fonte do datasets.yaml.")


def read_yolo(src: dict) -> tuple[list[Item], set[str]]:
    root = _p(src["root"])
    names = _names_for_yolo(src, root)
    items = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        parts = list(img.with_suffix(".txt").parts)
        cands = [img.with_suffix(".txt")]
        if "images" in parts:   # convenção images/ <-> labels/
            i = len(parts) - 1 - parts[::-1].index("images")
            cands.insert(0, Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt"))
        label = next((c for c in cands if c.exists()), None)
        if label is None:
            continue
        boxes = []
        for line in label.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            k = int(float(v[0]))
            if k >= len(names):
                continue
            nums = [float(x) for x in v[1:]]
            if len(nums) > 4:   # polígono (segmentação) -> caixa envolvente
                xs, ys = nums[0::2], nums[1::2]
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                nums = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
            boxes.append((names[k], *nums[:4]))
        if boxes:
            items.append(Item(img, boxes))
    return items, set(names)


# ----------------------------------------------------------------------------- principal
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "training" / "datasets.yaml"))
    ap.add_argument("--materials-dir", default=None, help="(testes) pasta com materials.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mats = MaterialsCfg.load(Path(args.materials_dir)) if args.materials_dir else MaterialsCfg.load()
    class_ids = {m.id: i for i, m in enumerate(mats)}
    out = _p(cfg.get("output_dir", "datasets/reciclagem"))
    rng = random.Random(cfg.get("seed", 42))
    min_area = float(cfg.get("min_box_area_frac", 0.0))

    pool: list[tuple[str, Item, list[tuple[int, float, float, float, float]]]] = []
    used_keys: set[str] = set()
    for src in cfg["sources"]:
        if not src.get("enabled", True):
            print(f"[{src['name']}] desativada (enabled: false)")
            continue
        try:
            items, src_classes = (read_coco if src["format"] == "coco" else read_yolo)(src)
        except FileNotFoundError as e:
            raise SystemExit(f"[{src['name']}] arquivo/pasta não encontrado: {e}\n"
                             "→ veja training/README.md (como baixar) ou desative a fonte em datasets.yaml.")
        cmap = {normalize_name(k): v for k, v in (src.get("class_map") or {}).items()}
        if src.get("identity"):
            cmap.update({normalize_name(c): normalize_name(c) for c in src_classes if normalize_name(c) in class_ids})
        unknown = sorted({v for v in cmap.values() if v not in class_ids})
        if unknown:
            raise SystemExit(f"[{src['name']}] class_map aponta para materiais inexistentes: {unknown}")
        kept = 0
        for it in items:
            mapped = []
            for name, cx, cy, w, h in it.boxes:
                tgt = cmap.get(normalize_name(name))
                if tgt is None or w * h < min_area:
                    continue
                used_keys.add(f"{src['name']}:{normalize_name(name)}")
                mapped.append((class_ids[tgt], cx, cy, w, h))
            if mapped:
                pool.append((src["name"], it, mapped))
                kept += 1
        ignored = sorted(c for c in src_classes if normalize_name(c) not in cmap)
        dead = sorted(k for k in cmap if f"{src['name']}:{k}" not in used_keys)
        print(f"[{src['name']}] {len(items)} imagens lidas, {kept} com ao menos 1 objeto mapeado")
        if ignored:
            print(f"    classes de origem IGNORADAS ({len(ignored)}): {', '.join(ignored[:12])}{'…' if len(ignored) > 12 else ''}")
        if dead:
            print(f"    chaves do class_map sem nenhum objeto encontrado (nome diferente no seu download?): {dead}")

    if not pool:
        raise SystemExit("Nenhuma imagem utilizável. Baixe os datasets (training/README.md) e confira datasets.yaml.")

    # split por imagem, embaralhado e reprodutível
    rng.shuffle(pool)
    n = len(pool)
    n_test, n_val = int(n * cfg.get("test_fraction", 0.1)), int(n * cfg.get("val_fraction", 0.15))
    splits = {"test": pool[:n_test], "val": pool[n_test:n_test + n_val], "train": pool[n_test + n_val:]}

    if out.exists():
        shutil.rmtree(out)
    stats = {s: Counter() for s in splits}
    for split, rows in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for k, (srcname, it, boxes) in enumerate(rows):
            stem = f"{srcname}_{k:06d}"
            dst = out / "images" / split / f"{stem}{it.image.suffix.lower()}"
            if cfg.get("link_images"):
                dst.symlink_to(it.image.resolve())
            else:
                shutil.copy2(it.image, dst)
            with open(out / "labels" / split / f"{stem}.txt", "w") as f:
                for c, cx, cy, w, h in boxes:
                    f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    stats[split][c] += 1

    names = {i: m.id for i, m in enumerate(mats)}
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
         "names": names}, allow_unicode=True, sort_keys=False), encoding="utf-8")

    report = {"images": {s: len(r) for s, r in splits.items()},
              "instances": {s: {names[c]: v for c, v in sorted(cnt.items())} for s, cnt in stats.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDataset pronto em {out}  ({report['images']})")
    print(f"{'classe':<22}{'treino':>8}{'val':>7}{'teste':>7}")
    weak = []
    for c, name in names.items():
        tr, va, te = (stats[s][c] for s in ("train", "val", "test"))
        print(f"{name:<22}{tr:>8}{va:>7}{te:>7}")
        if tr < 150:
            weak.append((name, tr))
    if weak:
        print("\n⚠ Classes com poucas amostras de treino (o modelo tende a ir mal nelas):")
        for name, tr in weak:
            print(f"   - {name}: {tr} instâncias → junte mais dados (outros datasets ou suas próprias fotos)")
    print(f"\nPróximo passo:  python training/train.py --data {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
