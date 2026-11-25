import time

from recicla.config import MaterialsCfg, Settings
from recicla.core.engine import Engine
from recicla.core.session import Session
from tests.fakes import FakeTracker, make_video


def _engine(tmp_path, frames=90):
    video = make_video(tmp_path / "demo.avi", frames=frames)
    st = Settings.load()
    st.camera.source = str(video)
    st.model.max_infer_fps = 120
    mats = MaterialsCfg.load()
    ses = Session(mats)
    return Engine(st, mats, ses, tracker_factory=FakeTracker), ses


def _wait(cond, timeout=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_pipeline_counts_each_object_once_and_stops_cleanly(tmp_path):
    eng, ses = _engine(tmp_path)
    assert eng.start()
    assert _wait(lambda: ses.snapshot().total >= 2)
    time.sleep(1.5)                                   # vídeo em loop: IDs repetidos não podem recontar
    snap = ses.snapshot()
    assert snap.total == 2, snap
    assert snap.by_material["garrafa_pet"].count == 1 and snap.by_material["lata_aluminio"].count == 1
    res = eng.latest_result()
    assert res is not None and res.frame.shape[2] == 3 and len(res.detections) >= 2
    assert any(d.state == "counted" for d in res.detections)
    assert any(d.state == "untracked" for d in res.detections) or res.frame_id < 40
    eng.pause()
    assert eng.state == "paused"
    time.sleep(0.2)                                   # deixa terminar o quadro que já estava em processamento
    fid = eng.latest_result().frame_id
    time.sleep(0.5)
    assert eng.latest_result().frame_id == fid       # pausado: não processa
    eng.resume()
    assert _wait(lambda: eng.latest_result().frame_id > fid)
    eng.stop()
    assert eng.state == "idle" and eng.camera is None and eng.latest_result() is None


def test_restart_counting_recounts_objects_in_view(tmp_path):
    eng, ses = _engine(tmp_path)
    eng.start()
    assert _wait(lambda: ses.snapshot().total >= 2)
    ses.reset()
    eng.restart_counting()
    assert _wait(lambda: ses.snapshot().total >= 2)   # continuam visíveis -> contam de novo na nova sessão
    eng.stop()


def test_engine_reuses_loaded_model_between_runs(tmp_path):
    eng, ses = _engine(tmp_path)
    eng.start()
    assert _wait(lambda: eng.tracker is not None and eng.latest_result() is not None)
    first = eng.tracker
    eng.stop()
    eng.start()
    assert _wait(lambda: eng.latest_result() is not None)
    assert eng.tracker is first and first.resets >= 1   # mesmo modelo, tracker zerado
    eng.stop()


def test_bad_camera_reports_error_without_crashing(tmp_path):
    st = Settings.load()
    st.camera.source = str(tmp_path / "nao_existe.mp4")
    mats = MaterialsCfg.load()
    eng = Engine(st, mats, Session(mats), tracker_factory=FakeTracker)
    assert eng.start() is False and eng.state == "idle"
    kind, (lvl, msg) = eng.events.get_nowait()
    assert kind == "status" and lvl == "error" and "não_existe" not in msg and "Não foi possível abrir" in msg


def test_inference_errors_are_survived_then_reported(tmp_path):
    eng, ses = _engine(tmp_path)

    class Flaky(FakeTracker):
        def infer(self, frame):
            self.n += 1
            if self.n % 2:                      # falha intermitente: deve sobreviver
                raise RuntimeError("boom")
            return super().infer(frame)
    eng._factory = Flaky
    eng.start()
    assert _wait(lambda: eng.latest_result() is not None)
    eng.stop()

    class Broken(FakeTracker):
        def infer(self, frame):
            raise RuntimeError("sempre falha")
    eng2, _ = _engine(tmp_path)
    eng2._factory = Broken
    eng2.start()
    assert _wait(lambda: any(k == "error" for k, _ in list(eng2.events.queue)))
    eng2.stop()
