import csv
import time

from recicla.config import MaterialsCfg, Settings
from recicla.core.estimates import fmt_money, fmt_weight, group_estimate
from recicla.core.session import Session
from recicla.core.types import CountEvent


def test_default_materials_and_lookup():
    m = MaterialsCfg.load()
    assert "garrafa_pet" in m.materials and len(m.materials) == 7
    assert m.lookup_class("Garrafa PET").id == "garrafa_pet"   # normalização tolerante
    assert m.lookup_class("garrafa_pet").id == "garrafa_pet"
    assert m.lookup_class("lata-aluminio").id == "lata_aluminio"  # normalização
    assert m.lookup_demo("bottle").id == "garrafa_plastica"
    assert m.lookup_demo("person") is None


def test_estimates_math():
    m = MaterialsCfg.load().get("lata_aluminio")
    kg, val = group_estimate(m, 100)
    assert abs(kg - 100 * m.avg_weight_g / 1000) < 1e-9
    assert abs(val - kg * m.price_per_kg) < 1e-9
    assert fmt_money(1234.5) == "R$ 1.234,50"
    assert fmt_weight(0.014) == "14 g" and fmt_weight(2.5) == "2,50 kg"


def test_session_totals_reset_and_rate():
    mats = MaterialsCfg.load()
    s = Session(mats)
    now = time.time()
    for i in range(3):
        s.add_event(CountEvent(i + 1, "garrafa_pet", 0.9, now))
    s.add_event(CountEvent(9, "lata_aluminio", 0.8, now))
    snap = s.snapshot()
    assert snap.total == 4 and snap.by_material["garrafa_pet"].count == 3
    assert snap.rate_last_min == 4
    expected_kg = 3 * mats.get("garrafa_pet").avg_weight_g / 1000 + mats.get("lata_aluminio").avg_weight_g / 1000
    assert abs(snap.total_weight_kg - expected_kg) < 1e-9
    assert [e.track_id for _, e in s.events_since(2)] == [3, 9]
    s.reset()
    assert s.snapshot().total == 0 and not s.has_data()


def test_csv_export_and_history(tmp_path):
    s = Session(MaterialsCfg.load())
    s.add_event(CountEvent(1, "papel", 0.77, time.time()))
    ev, sm = s.export_csv(tmp_path)
    rows = list(csv.reader(open(ev, encoding="utf-8-sig"), delimiter=";"))
    assert len(rows) == 2 and rows[1][2] == "papel"
    assert "ESTIMATIVA" in rows[0][-1]
    summary = open(sm, encoding="utf-8-sig").read()
    assert "TOTAL" in summary and "ESTIMATIVAS" in summary
    assert s.append_history_line(tmp_path).exists()


def test_settings_roundtrip_with_local_override(tmp_path):
    import shutil
    from recicla.paths import CONFIG_DIR
    for n in ("settings.yaml", "materials.yaml"):
        shutil.copy(CONFIG_DIR / n, tmp_path / n)
    st = Settings.load(tmp_path)
    assert st.camera.source == 1 and st.model.imgsz == 640
    st.model.conf = 0.55
    st.camera.source = "/tmp/video.mp4"
    st.save(tmp_path)
    st2 = Settings.load(tmp_path)
    assert st2.model.conf == 0.55 and st2.camera.source == "/tmp/video.mp4"
    mats = MaterialsCfg.load(tmp_path)
    mats.get("papel").price_per_kg = 1.23
    mats.save(tmp_path)
    assert MaterialsCfg.load(tmp_path).get("papel").price_per_kg == 1.23
    assert MaterialsCfg.load(tmp_path).get("papel").label == "Papel"   # resto vem do padrão
