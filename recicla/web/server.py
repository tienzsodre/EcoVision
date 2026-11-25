"""Interface web local para o motor de triagem."""
import io
import logging
import math
import os
import queue
import threading
import webbrowser
from dataclasses import asdict, replace
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file
from markupsafe import Markup
from PIL import Image
from werkzeug.serving import make_server

from ..config import MaterialsCfg, Settings, parse_source
from ..core.engine import Engine
from ..core.session import Session
from ..paths import ROOT, resolve
from ..ui.overlay import draw_detections

ACTIONS = {"start", "pause", "resume", "stop", "reset", "new", "settings", "class_override"}
WEB_DIR = Path(__file__).resolve().parent


class WebInterface:
    def __init__(self, settings: Settings, materials: MaterialsCfg):
        self.settings = settings
        self.materials = materials
        self.session = Session(materials)
        self.engine = Engine(settings, materials, self.session)
        self.status = "Pronto para iniciar"
        self.error = None
        self.startup = "loading"
        self.startup_error = None
        self.commands = queue.Queue(maxsize=32)
        self.session_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.cached_result = None
        self.cached_jpeg = None

    def create_app(self) -> Flask:
        app = Flask(__name__, template_folder=str(WEB_DIR),
                    static_folder=str(WEB_DIR / "static"))
        app.add_url_rule("/", view_func=self.index)
        app.add_url_rule("/ecovision.svg", view_func=self.logo)
        app.add_url_rule("/api/state", view_func=self.state)
        app.add_url_rule("/api/frame", view_func=self.frame)
        app.add_url_rule("/api/action", view_func=self.action, methods=["POST"])
        app.add_url_rule("/api/export", view_func=self.export)
        return app

    def index(self):
        # O SVG do projeto é um recurso local confiável, inserido inline para animação.
        return render_template("index.html", logo=Markup(self.logo_path().read_text(encoding="utf-8")))

    @staticmethod
    def logo_path():
        official = ROOT / "ecovision.svg"
        return official if official.is_file() else WEB_DIR / "static" / "provisional-logo.svg"

    def logo(self):
        return send_file(self.logo_path(), mimetype="image/svg+xml")

    def state(self):
        result = self.engine.latest_result()
        tracker = self.engine.tracker
        model = asdict(tracker.info) if tracker and tracker.info else None
        if model:
            model["device"] = tracker.device
        with self.session_lock:
            stats = self.session.snapshot()
            events = self.session.events_since(max(0, stats.last_seq - 50))
        return jsonify(
            state=self.engine.state, status=self.status, model=model, stats=asdict(stats),
            startup=self.startup, startup_error=self.startup_error,
            error=self.error,
            class_override=self.engine.class_override,
            materials=[asdict(m) for m in self.materials], settings=asdict(self.settings),
            fps=result.infer_fps if result else 0, ms=result.infer_ms if result else 0,
            events=[asdict(event) for _, event in events[-50:]],
        )

    def frame(self):
        headers = {"Cache-Control": "no-store"}
        with self.frame_lock:
            result = self.engine.latest_result()
            if result is None:
                self.cached_result = self.cached_jpeg = None
                return Response(status=204, headers=headers)
            if self.cached_result is not result:
                image = Image.fromarray(cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB))
                draw_detections(image, result.detections, 1, self.materials)
                with io.BytesIO() as buffer:
                    image.save(buffer, format="JPEG", quality=82)
                    self.cached_jpeg = buffer.getvalue()
                self.cached_result = result
            return Response(self.cached_jpeg, mimetype="image/jpeg", headers=headers)

    def action(self):
        data = request.get_json(silent=True)
        action = data.get("action") if isinstance(data, dict) else None
        if not isinstance(action, str) or action not in ACTIONS:
            return jsonify(error="Ação inválida."), 400
        if self.startup == "loading":
            return jsonify(error="Aguarde a inicialização do modelo."), 503
        if action == "class_override":
            if "material" not in data or (data["material"] is not None and not isinstance(data["material"], str)):
                return jsonify(error="Classificação manual inválida."), 400
            try:
                self.engine.set_class_override(data["material"])
            except ValueError as exc:
                return jsonify(error=str(exc)), 400
            return jsonify(ok=True)
        if action == "settings":
            try:
                source, path = data["source"], data["path"]
                conf = float(data["conf"])
                if (type(source) not in (str, int) or not str(source).strip()
                        or not isinstance(path, str) or not path.strip()
                        or not math.isfinite(conf) or not 0.01 <= conf <= 0.99):
                    raise ValueError
                data = dict(action=action, source=parse_source(source), path=path.strip(), conf=conf)
            except (KeyError, ValueError, TypeError, OverflowError):
                return jsonify(error="Informe uma fonte, um modelo e confiança entre 0,01 e 0,99."), 400
        try:
            self.commands.put_nowait(data)
        except queue.Full:
            return jsonify(error="Aguarde os comandos em andamento."), 503
        return jsonify(ok=True), 202

    def export(self):
        with self.session_lock:
            if not self.session.has_data():
                return jsonify(error="Sem detecções para exportar."), 400
            _, summary = self.session.export_csv()
            contents = summary.read_bytes()
        return send_file(io.BytesIO(contents), mimetype="text/csv",
                         download_name=summary.name, as_attachment=True)

    def process_command(self, data):
        action = data["action"]
        if action in ("start", "pause", "resume", "stop"):
            self.error = None
            getattr(self.engine, action)()
            if action == "stop":
                self.status = "Câmera encerrada."
        elif action in ("reset", "new"):
            with self.session_lock:
                self.engine.reset_session(archive=(action == "new"))
        elif self.engine.state != "idle":
            self.status = "Encerre a câmera antes de alterar as configurações."
        else:
            self.error = None
            updated = replace(
                self.settings, camera=replace(self.settings.camera, source=data["source"]),
                model=replace(self.settings.model, path=data["path"], conf=data["conf"]),
            )
            updated.save()
            self.settings.camera = updated.camera
            self.settings.model = updated.model
            self.engine.invalidate_model()
            self.status = "Configurações salvas."

    def prepare_startup(self):
        ready = self.engine.prepare_model()
        self.engine.events.put(("startup", ready))

    def process_events(self):
        while True:
            try:
                kind, payload = self.engine.events.get_nowait()
            except queue.Empty:
                return
            if kind == "startup":
                if payload:
                    self.status = " ".join(self.engine.tracker.info.warnings) or "Pronto para iniciar"
                    self.startup = "ready"
                else:
                    self.startup_error = self.status
                    self.startup = "error"
            elif kind == "status":
                self.status = payload[1]
                if payload[0] == "error":
                    self.error = self.status
            elif kind == "error":
                self.status = str(payload)
                self.error = self.status
            elif kind == "stopped":
                self.engine.stop()
            elif kind == "state" and payload == "running":
                tracker = self.engine.tracker
                warnings = tracker.info.warnings if tracker and tracker.info else []
                self.status = " ".join(warnings) or "Detecção em andamento."
            elif kind == "state" and payload == "paused":
                self.status = "Detecção pausada."

    def serve(self, open_browser: bool = True):
        server = make_server("127.0.0.1", 0, self.create_app(), threaded=True)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"EcoVision: {url} | Ctrl+C para encerrar", flush=True)
        try:
            threading.Thread(target=self.prepare_startup, name="model-startup", daemon=True).start()
            if open_browser:
                webbrowser.open(url)
            # O AVFoundation precisa abrir a câmera na thread principal.
            while True:
                self.process_events()
                try:
                    self.process_command(self.commands.get(timeout=0.05))
                except queue.Empty:
                    continue
                except Exception as exc:
                    self.status = str(exc)
                    self.error = self.status
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server_thread.join()
            server.server_close()
            self.engine.stop()
            with self.session_lock:
                self.session.append_history_line()


def run(open_browser: bool = True):
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "config" / "ultralytics.local"))
    settings = Settings.load()
    if not resolve(settings.model.fallback).exists() and (ROOT / "yolo11n.pt").is_file():
        settings.model.fallback = "yolo11n.pt"
    WebInterface(settings, MaterialsCfg.load()).serve(open_browser)
