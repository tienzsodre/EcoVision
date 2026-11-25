"""Smoke test visual da interface sob display virtual (xvfb). Salva capturas de tela em /tmp."""
import sys, time
sys.path.insert(0, ".")
from PIL import ImageGrab
from recicla.config import MaterialsCfg, Settings
from recicla.core.engine import Engine
from recicla.ui.app import MainWindow
from tests.fakes import FakeTracker, make_video

video = make_video("/tmp/ui_demo.avi", frames=120, size=(640, 360))
st = Settings.load(); st.camera.source = video; st.model.max_infer_fps = 30
mats = MaterialsCfg.load()
win = MainWindow(st, mats, engine_factory=lambda s, m, ses: Engine(s, m, ses, tracker_factory=FakeTracker))

def pump(sec):
    t0 = time.time()
    while time.time() - t0 < sec:
        win.update_idletasks(); win.update(); time.sleep(0.01)

pump(0.5)
ImageGrab.grab().save("/tmp/ui_idle.png")
win._start(); pump(4.0)
# força alguns eventos "contados" extras para ver histórico/estatísticas preenchidos
import time as _t
from recicla.core.types import CountEvent
for i, mid in enumerate(["garrafa_pet", "lata_aluminio", "caixa_papelao", "papel", "embalagem_plastica", "lata_aco", "garrafa_plastica", "garrafa_pet"]):
    win.session.add_event(CountEvent(100 + i, mid, 0.6 + i * 0.04, _t.time()))
pump(1.0)
ImageGrab.grab().save("/tmp/ui_running.png")
win._toggle_pause(); pump(0.6)
print("state after pause:", win._state, "| btn text:", win._btn["pause"].cget("text"))
win._open_settings(); pump(1.0)
ImageGrab.grab().save("/tmp/ui_settings.png")
for w in win.winfo_children():
    if w.winfo_class() == "Toplevel": w.destroy()
pump(0.3)
win.geometry("1200x740"); pump(1.0)
ImageGrab.grab().save("/tmp/ui_small.png")
snap = win.session.snapshot()
print("total:", snap.total, "| history rows:", len(win.history.tree.get_children()))
win._stop(); pump(0.5)
print("state after stop:", win._state)
win.engine.stop(); win.destroy()
print("OK")
