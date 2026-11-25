"""
Smoke test do PIPELINE de treino (prepare -> train -> evaluate -> export) com dados SINTÉTICOS
(retângulos/círculos coloridos). Serve só para provar que os scripts rodam de ponta a ponta;
o modelo resultante NÃO sabe reconhecer recicláveis de verdade.
"""
import random, subprocess, sys
from pathlib import Path
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
W = Path("/tmp/smoke"); raw = W / "raw"; (raw / "images").mkdir(parents=True, exist_ok=True); (raw / "labels").mkdir(exist_ok=True)
(raw / "classes.txt").write_text("garrafa_pet\nlata_aluminio\n")
rnd = random.Random(0)
for i in range(80):
    im = Image.new("RGB", (320, 320), (rnd.randint(40, 90),) * 3); d = ImageDraw.Draw(im); lines = []
    for _ in range(rnd.randint(1, 3)):
        c = rnd.randint(0, 1); w, h = (50, 110) if c == 0 else (70, 70)
        x, y = rnd.randint(0, 320 - w), rnd.randint(0, 320 - h)
        (d.rectangle if c == 0 else d.ellipse)([x, y, x + w, y + h], fill=(40, 140, 255) if c == 0 else (240, 240, 240))
        lines.append(f"{c} {(x + w / 2) / 320:.5f} {(y + h / 2) / 320:.5f} {w / 320:.5f} {h / 320:.5f}")
    im.save(raw / "images" / f"s{i}.jpg"); (raw / "labels" / f"s{i}.txt").write_text("\n".join(lines))
(W / "cfg.yaml").write_text(yaml.safe_dump({"output_dir": str(W / "ds"), "val_fraction": 0.2, "test_fraction": 0.1,
    "sources": [{"name": "sint", "format": "yolo", "root": str(raw), "identity": True}]}))
run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, check=True)
run("training/prepare_dataset.py", "--config", str(W / "cfg.yaml"))
run("training/train.py", "--data", str(W / "ds/data.yaml"), "--base", "models/yolo11n.pt", "--epochs", "2", "--imgsz", "160",
    "--batch", "8", "--workers", "0", "--device", "cpu", "--project", str(W / "runs/detect"), "--name", "smoke")
run("training/evaluate.py", "--weights", str(W / "runs/detect/smoke/weights/best.pt"), "--data", str(W / "ds/data.yaml"),
    "--imgsz", "160", "--device", "cpu", "--name", "smoke")
run("training/export_model.py", "--weights", str(W / "runs/detect/smoke/weights/best.pt"), "--dest", str(W / "export/smoke_best.pt"))
print("SMOKE TRAINING PIPELINE OK:", list((W / "export").glob("*")))
