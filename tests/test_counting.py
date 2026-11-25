from recicla.core.counting import CountingManager, iou
from recicla.core.types import RawTrack


def T(i, mat="garrafa_pet", conf=0.9, box=(100, 100, 200, 200)):
    return RawTrack(i, mat, conf, box)


def run(cm, frames, t0=0.0, dt=0.033):
    events = []
    for k, tracks in enumerate(frames):
        _, ev = cm.update(tracks, now=t0 + k * dt)
        events += ev
    return events


def test_iou_basic():
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0.3 < iou((0, 0, 10, 10), (5, 0, 15, 10)) < 0.4


def test_counts_only_after_min_hits_and_only_once():
    cm = CountingManager(min_hits=5)
    ev = run(cm, [[T(1)]] * 4)
    assert ev == []                      # ainda candidato
    ev = run(cm, [[T(1)]] * 100, t0=1.0)  # permanece 100 quadros na tela
    assert len(ev) == 1 and ev[0].track_id == 1


def test_short_flicker_is_not_counted():
    cm = CountingManager(min_hits=5)
    assert run(cm, [[T(7)]] * 3) == []


def test_class_vote_uses_weighted_majority():
    cm = CountingManager(min_hits=5)
    frames = [[T(1, "garrafa_plastica", 0.5)], [T(1, "garrafa_pet", 0.9)]] * 3
    ev = run(cm, frames)
    assert len(ev) == 1 and ev[0].material == "garrafa_pet"


def test_two_distinct_objects_counted_separately():
    cm = CountingManager(min_hits=3)
    frames = [[T(1, box=(0, 0, 50, 50)), T(2, "lata_aluminio", box=(300, 300, 360, 360))]] * 5
    ev = run(cm, frames)
    assert sorted(e.material for e in ev) == ["garrafa_pet", "lata_aluminio"]


def test_id_switch_same_place_is_not_double_counted():
    cm = CountingManager(min_hits=3, reid_window_s=3.0, reid_iou=0.4)
    ev = run(cm, [[T(1)]] * 5)                     # objeto contado com ID 1
    assert len(ev) == 1
    ev = run(cm, [[]] * 10 + [[T(2)]] * 5, t0=1.0)  # some 10 quadros, volta com ID 2, mesmo lugar
    assert ev == []                                 # herdado, não recontado


def test_class_override_keeps_counting_identity_and_original_votes():
    cm = CountingManager(min_hits=2)
    garrafa = T(1, "garrafa")
    assert cm.update([garrafa], now=0.0, class_override="garrafa")[1] == []
    dets, events = cm.update([garrafa], now=0.1, class_override="lata")
    assert dets[0].predicted_material == "garrafa"
    assert [(e.material, e.predicted_material, e.conf) for e in events] == [("lata", "garrafa", None)]
    assert cm.update([garrafa], now=0.2)[1] == []
    assert cm.update([T(2, "garrafa")], now=0.3, class_override="garrafa")[1] == []
    assert cm.update([T(2, "garrafa")], now=0.4, class_override="garrafa")[1] == []

    lata = T(3, "lata", box=(300, 300, 400, 400))
    cm.update([lata], now=0.5, class_override="garrafa")
    _, events = cm.update([lata], now=0.6)
    assert len(events) == 1 and events[0].material == "lata" and events[0].conf == lata.conf


def test_new_object_elsewhere_is_counted():
    cm = CountingManager(min_hits=3)
    run(cm, [[T(1)]] * 5)
    ev = run(cm, [[T(2, box=(500, 400, 600, 500))]] * 5, t0=1.0)
    assert len(ev) == 1


def test_reid_expires_after_window():
    cm = CountingManager(min_hits=3, reid_window_s=1.0)
    run(cm, [[T(1)]] * 5, t0=0.0)
    ev = run(cm, [[T(2)]] * 5, t0=1.5)   # 1,5 s depois > janela de 1 s
    assert len(ev) == 1


def test_reid_disabled():
    cm = CountingManager(min_hits=3, reid_window_s=0)
    run(cm, [[T(1)]] * 5)
    assert len(run(cm, [[T(2)]] * 5, t0=0.5)) == 1


def test_reid_requires_same_class():
    cm = CountingManager(min_hits=3)
    run(cm, [[T(1, "garrafa_pet")]] * 5)
    assert len(run(cm, [[T(2, "lata_aco")]] * 5, t0=0.5)) == 1


def test_untracked_detection_is_never_counted():
    cm = CountingManager(min_hits=1)
    dets, ev = cm.update([RawTrack(None, "papel", 0.9, (0, 0, 10, 10))], now=0.0)
    assert ev == [] and dets[0].state == "untracked"


def test_reset_forgets_everything_and_recounts():
    cm = CountingManager(min_hits=3)
    assert len(run(cm, [[T(1)]] * 5)) == 1
    cm.reset()
    assert len(run(cm, [[T(1)]] * 5, t0=10.0)) == 1


def test_stale_states_are_cleaned():
    cm = CountingManager(min_hits=3, reid_window_s=1.0)
    run(cm, [[T(i)] for i in range(1, 50)])
    cm.update([], now=100.0)
    assert len(cm._states) == 0
