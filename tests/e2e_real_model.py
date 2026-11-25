"""
Teste de ponta a ponta com o YOLO REAL (precisa de torch + ultralytics + internet p/ baixar yolo11n.pt).
Usa uma foto de ônibus com pessoas: mapeamos 'person'->papel e 'bus'->caixa_papelao SÓ neste teste,
para validar detector -> ByteTrack -> contagem -> sessão. (Não valida precisão em recicláveis.)
"""
import sys, time
sys.path.insert(0, ".")
import cv2, numpy as np
from recicla.config import MaterialsCfg, Settings
from recicla.core.detector import YoloTracker
from recicla.core.engine import Engine
from recicla.core.session import Session

img = cv2.imread("/tmp/bus.jpg")
H, W = img.shape[:2]

def shifted(i):
    M = np.float32([[1, 0, i * 2], [0, 1, 0]])
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

mats = MaterialsCfg.load()
mats.get("papel").demo_coco = ["person"]
mats.get("caixa_papelao").demo_coco = ["bus"]
mats._rebuild_index()
st = Settings.load()
st.model.max_infer_fps = 60

tr = YoloTracker(st.model, st.tracking, mats)
t0 = time.time(); info = tr.load(warmup_shape=(H, W)); print("load %.1fs" % (time.time() - t0), info.weights, info.device, "demo=", info.is_demo)
print("warnings:", info.warnings)
ids = {}
for i in range(12):
    t0 = time.perf_counter(); tracks = tr.infer(shifted(i)); dt = (time.perf_counter() - t0) * 1000
    for t in tracks: ids.setdefault(t.track_id, set()).add(t.material)
    print(f"frame {i:2d}: {len(tracks)} tracks, {dt:5.0f} ms, ids={sorted(k for k in ids if k is not None)}")
assert len(ids) >= 3 and None not in ids, ids
tr.reset_tracker()
t = tr.infer(shifted(0))
print("after reset ids:", sorted(x.track_id for x in t if x.track_id is not None))
assert min(x.track_id for x in t if x.track_id is not None) == 1, "reset_tracker não reiniciou os IDs"

# --- pipeline completo com vídeo
vw = cv2.VideoWriter("/tmp/bus.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10, (W, H))
for i in range(60): vw.write(shifted(i))
vw.release()
st.camera.source = "/tmp/bus.avi"; st.tracking.min_hits = 4
ses = Session(mats)
eng = Engine(st, mats, ses)          # fábrica real: YoloTracker
assert eng.start()
t0 = time.time()
while time.time() - t0 < 40 and ses.snapshot().total < 4: time.sleep(0.3)
time.sleep(4)                        # vídeo em loop: contagem não pode crescer por repetição dos mesmos objetos
snap = ses.snapshot()
res = eng.latest_result()
print("session total:", snap.total, {k: v.count for k, v in snap.by_material.items() if v.count}, "| infer_fps %.1f" % res.infer_fps)
eng.stop()
