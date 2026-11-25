import json
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _img(path, size=(200, 100)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 90, 90)).save(path)


def test_prepare_dataset_coco_and_yolo_sources(tmp_path):
    # --- fonte COCO (estilo TACO): 10 imagens, categorias com nomes "humanos"
    imgs = [{"id": i, "file_name": f"batch_1/{i:03d}.jpg", "width": 200, "height": 100} for i in range(10)]
    for im in imgs:
        _img(tmp_path / "taco" / im["file_name"])
    anns = []
    for i in range(10):
        anns.append({"id": 2 * i, "image_id": i, "category_id": 1, "bbox": [10, 10, 60, 40], "iscrowd": 0})
        anns.append({"id": 2 * i + 1, "image_id": i, "category_id": 2, "bbox": [100, 20, 50, 50], "iscrowd": 0})
    anns.append({"id": 99, "image_id": 0, "category_id": 3, "bbox": [0, 0, 5, 5], "iscrowd": 0})  # classe ignorada
    (tmp_path / "taco.json").write_text(json.dumps({
        "images": imgs, "annotations": anns,
        "categories": [{"id": 1, "name": "Clear plastic bottle"}, {"id": 2, "name": "Drink can"}, {"id": 3, "name": "Cigarette"}]}))

    # --- fonte YOLO com polígono e nomes via classes.txt
    for i in range(6):
        _img(tmp_path / "yolo" / "images" / f"y{i}.jpg")
        (tmp_path / "yolo" / "labels").mkdir(exist_ok=True)
        (tmp_path / "yolo" / "labels" / f"y{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.2 0.3 0.2 0.3 0.4\n")
    (tmp_path / "yolo" / "classes.txt").write_text("papel\nglass\n")

    cfg = {"output_dir": str(tmp_path / "out"), "val_fraction": 0.2, "test_fraction": 0.1, "seed": 1,
           "sources": [
               {"name": "taco", "format": "coco", "images_root": str(tmp_path / "taco"), "annotations": str(tmp_path / "taco.json"),
                "class_map": {"clear plastic bottle": "garrafa_pet", "DRINK-CAN": "lata_aluminio", "nome inexistente": "papel"}},
               {"name": "meu", "format": "yolo", "root": str(tmp_path / "yolo"), "identity": True}]}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, str(ROOT / "training" / "prepare_dataset.py"), "--config", str(tmp_path / "cfg.yaml")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "classes de origem IGNORADAS" in r.stdout and "Cigarette" in r.stdout
    assert "nome_inexistente" in r.stdout     # chave morta do class_map é denunciada

    out = tmp_path / "out"
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"][0] == "garrafa_pet" and len(data["names"]) == 7
    n_img = {s: len(list((out / "images" / s).glob("*"))) for s in ("train", "val", "test")}
    assert sum(n_img.values()) == 16 and n_img["train"] > n_img["val"] >= 1
    n_lbl = {s: len(list((out / "labels" / s).glob("*.txt"))) for s in ("train", "val", "test")}
    assert n_lbl == n_img
    classes_used, boxes = set(), 0
    for f in (out / "labels").rglob("*.txt"):
        for line in f.read_text().splitlines():
            c, *vals = line.split()
            classes_used.add(data["names"][int(c)])
            assert all(0.0 <= float(v) <= 1.0 for v in vals) and len(vals) == 4
            boxes += 1
    # papel (yolo identity) + garrafa_pet + lata_aluminio; "glass" e "Cigarette" ficaram de fora
    assert classes_used == {"garrafa_pet", "lata_aluminio", "papel"}
    assert boxes == 10 * 2 + 6 * 1
