"""Janela principal (CustomTkinter). Toda a lógica pesada vive em recicla.core.* (outras threads)."""
from __future__ import annotations

import queue
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from ..config import MaterialsCfg, Settings
from ..core.engine import Engine
from ..core.session import Session
from ..paths import SESSIONS_DIR
from . import overlay
from . import theme as T
from .panels import HistoryPanel, StatsPanel, chip
from .settings_dialog import SettingsDialog

DISCLAIMER = "ATENÇÃO: peso e valor são ESTIMATIVAS (contagem × peso médio × preço/kg configuráveis), não medições reais."


def make_fonts(root) -> dict:
    avail = set(tkfont.families(root))
    family = next((f for f in ("Helvetica Neue", "SF Pro Text", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans")
                   if f in avail), "TkDefaultFont")
    F = lambda size, weight="normal": ctk.CTkFont(family=family, size=size, weight=weight)
    return {"family": family, "title": F(19, "bold"), "huge": F(40, "bold"), "big": F(20, "bold"),
            "body": F(13), "body_b": F(14, "bold"), "small": F(12), "small_b": F(12, "bold"),
            "tiny": F(11), "tiny_b": F(11, "bold"), "btn": F(13, "bold")}


class MainWindow(ctk.CTk):
    def __init__(self, settings: Settings, materials: MaterialsCfg, engine_factory=None):
        super().__init__(fg_color=T.BG)
        ctk.set_appearance_mode("dark")
        self.settings, self.materials = settings, materials
        self.session = Session(materials)
        self.engine = (engine_factory or Engine)(settings, materials, self.session)
        self.fonts = make_fonts(self)

        self.title("Triagem Inteligente de Recicláveis")
        self.geometry("1400x860")
        self.minsize(1280, 720)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._photo = None
        self._img_item = None
        self._last_result = None
        self._last_frame_id = -1
        self._hist_seq = 0
        self._last_stats = 0.0
        self._state = "idle"
        self._model_info = None
        self._clock_text = ""
        self._build()
        self._set_state("idle")
        self._poll()

    # ================================================================== construção
    def _build(self) -> None:
        f = self.fonts
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0, minsize=450)
        self.rowconfigure(1, weight=1)

        # ---- cabeçalho
        head = ctk.CTkFrame(self, fg_color=T.PANEL, corner_radius=0, height=58, border_width=0)
        head.grid(row=0, column=0, columnspan=2, sticky="ew")
        head.grid_propagate(False)
        ctk.CTkLabel(head, text="TRIAGEM INTELIGENTE DE RECICLÁVEIS", font=f["title"], text_color=T.TEXT).pack(side="left", padx=18)
        ctk.CTkLabel(head, text="detecção · rastreamento · contagem em tempo real (IA local)", font=f["small"],
                     text_color=T.MUTED).pack(side="left", padx=4)
        self.clock = chip(head, "--:--:--", font=f["small_b"])
        self.clock.pack(side="right", padx=(6, 18))
        self.device_chip = chip(head, "DISPOSITIVO —", font=f["tiny_b"])
        self.device_chip.pack(side="right", padx=6)
        self.model_chip = chip(head, "MODELO —", font=f["tiny_b"])
        self.model_chip.pack(side="right", padx=6)

        # ---- coluna esquerda: vídeo + controles
        left = ctk.CTkFrame(self, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(14, 7), pady=(14, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        vid = ctk.CTkFrame(left, fg_color=T.PANEL, corner_radius=T.RADIUS, border_width=1, border_color=T.BORDER)
        vid.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(vid, bg="#05080c", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self._placeholder = self.canvas.create_text(0, 0, text="Câmera desligada\nClique em  Iniciar  para começar",
                                                    fill=T.MUTED, font=(f["family"], 17), justify="center")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        ctrl = ctk.CTkFrame(left, fg_color=T.PANEL, corner_radius=T.RADIUS, border_width=1, border_color=T.BORDER)
        ctrl.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self._btn = {}
        specs = [("start", "Iniciar", T.OK, "#06210f", self._start),
                 ("pause", "Pausar", T.WARN, "#2b1a00", self._toggle_pause),
                 ("stop", "Encerrar", T.DANGER, "#2b0707", self._stop)]
        for i, (key, text, color, fg, cmd) in enumerate(specs):
            b = ctk.CTkButton(ctrl, text=text, command=cmd, fg_color=color, hover_color=color, text_color=fg,
                              text_color_disabled="#3a4658", width=118 if key == "start" else 100, height=36, font=f["btn"], corner_radius=10)
            b.grid(row=0, column=i, padx=(14 if i == 0 else 6, 0), pady=(12, 6))
            self._btn[key] = b
        ctk.CTkFrame(ctrl, width=2, height=26, fg_color=T.BORDER).grid(row=0, column=3, padx=12)
        for i, (key, text, cmd, wd) in enumerate([("new", "Nova sessão", self._new_session, 104), ("reset", "Zerar", self._reset, 64),
                                                  ("csv", "Exportar CSV", self._export, 104), ("cfg", "Configurações", self._open_settings, 112)]):
            b = ctk.CTkButton(ctrl, text=text, command=cmd, fg_color=T.PANEL_ALT, hover_color=T.BORDER, text_color=T.TEXT,
                              width=wd, height=36, font=f["btn"], corner_radius=10)
            b.grid(row=0, column=4 + i, padx=(0 if i else 0, 6), pady=(12, 6))
            self._btn[key] = b
        conf_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        conf_row.grid(row=1, column=0, columnspan=8, sticky="ew", padx=14, pady=(0, 12))
        self.conf_lbl = ctk.CTkLabel(conf_row, text="", font=f["small_b"], text_color=T.TEXT, width=190, anchor="w")
        self.conf_lbl.pack(side="left")
        self.conf_slider = ctk.CTkSlider(conf_row, from_=0.05, to=0.95, number_of_steps=90, width=260,
                                         progress_color=T.ACCENT, button_color=T.ACCENT, button_hover_color="#67e8f9",
                                         command=self._on_conf)
        self.conf_slider.set(self.settings.model.conf)
        self.conf_slider.pack(side="left", padx=6)
        self._on_conf(self.settings.model.conf)
        ctk.CTkLabel(conf_row, text="menor = detecta mais (e erra mais) · maior = só o que tem certeza",
                     font=f["tiny"], text_color=T.MUTED).pack(side="left", padx=12)

        # ---- coluna direita: estatísticas + monitoramento
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew", padx=(7, 14), pady=(14, 8))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self.stats = StatsPanel(right, self.materials, f)
        self.stats.grid(row=0, column=0, sticky="ew")
        self.history = HistoryPanel(right, self.materials, f, self.settings.ui.history_rows)
        self.history.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        # ---- rodapé
        foot = ctk.CTkFrame(self, fg_color=T.PANEL, corner_radius=0, height=34)
        foot.grid(row=2, column=0, columnspan=2, sticky="ew")
        foot.grid_propagate(False)
        self.status_lbl = ctk.CTkLabel(foot, text="Pronto.", font=f["small"], text_color=T.MUTED, anchor="w")
        self.status_lbl.pack(side="left", padx=18)
        ctk.CTkLabel(foot, text=DISCLAIMER, font=f["small"], text_color=T.WARN, anchor="e").pack(side="right", padx=18)

    # ================================================================== estado / botões
    def _set_state(self, state: str) -> None:
        self._state = state
        en = lambda ok: "normal" if ok else "disabled"
        self._btn["start"].configure(state=en(state == "idle"))
        self._btn["pause"].configure(state=en(state in ("running", "paused")),
                                     text="Retomar" if state == "paused" else "Pausar")
        self._btn["stop"].configure(state=en(state != "idle"))
        if state == "idle":
            self._clear_video()

    def _set_status(self, text: str, level: str = "info") -> None:
        color = {"info": T.MUTED, "warn": T.WARN, "error": T.DANGER, "ok": T.OK}.get(level, T.MUTED)
        self.status_lbl.configure(text=text, text_color=color)

    # ================================================================== ações
    def _start(self) -> None:
        self._set_status("Abrindo câmera…")
        self.update_idletasks()
        if not self.engine.start():
            self._drain_events()  # mostra o motivo (mensagem de erro da câmera)
            self._set_state("idle")

    def _toggle_pause(self) -> None:
        (self.engine.resume if self._state == "paused" else self.engine.pause)()

    def _stop(self) -> None:
        self.engine.stop()
        self._set_status("Câmera encerrada.")

    def _new_session(self) -> None:
        if self.session.has_data():
            ev, _ = self.session.export_csv()
            self.session.append_history_line()
            self._set_status(f"Sessão anterior salva em {ev.parent.name}/ ({ev.stem}). Nova sessão iniciada.", "ok")
        else:
            self._set_status("Nova sessão iniciada.", "ok")
        self._clear_session()

    def _reset(self) -> None:
        if self.session.has_data() and not messagebox.askyesno(
                "Zerar estatísticas", "Zerar a contagem atual?\nOs dados desta sessão serão perdidos "
                "(use “Exportar CSV” antes, se quiser guardá-los).", parent=self):
            return
        self._clear_session()
        self._set_status("Estatísticas zeradas.", "ok")

    def _clear_session(self) -> None:
        self.session.reset()
        self.engine.restart_counting()
        self.history.clear()
        self._hist_seq = 0
        self._last_stats = 0.0

    def _export(self) -> None:
        if not self.session.has_data():
            messagebox.showinfo("Exportar CSV", "Ainda não há detecções nesta sessão.", parent=self)
            return
        ev, sm = self.session.export_csv()
        self._set_status(f"CSV exportado: {ev}  e  {sm.name}", "ok")

    def _open_settings(self) -> None:
        SettingsDialog(self, self.settings, self.materials, self.fonts, self._settings_saved)

    def _settings_saved(self, old: Settings, new: Settings) -> None:
        self.engine.apply_live_settings()
        self.conf_slider.set(new.model.conf)
        self._on_conf(new.model.conf)
        self.stats.refresh_materials(self.materials)
        self.history.refresh_materials(self.materials)
        self.session.materials = self.materials
        needs_restart = (old.camera != new.camera or old.model.path != new.model.path
                         or old.model.device != new.model.device)
        if needs_restart:
            if self._state != "idle" and messagebox.askyesno(
                    "Reiniciar câmera", "As alterações de câmera/modelo só valem em uma nova partida.\n"
                    "Encerrar a câmera agora?", parent=self):
                self.engine.stop()
            if self._state == "idle" or self.engine.state == "idle":
                self.engine.invalidate_model()
        self._set_status("Configurações salvas.", "ok")

    def _on_conf(self, value: float) -> None:
        self.conf_lbl.configure(text=f"Confiança mínima: {float(value) * 100:.0f}%")
        self.engine.set_conf(value)

    def _on_close(self) -> None:
        try:
            self.engine.stop()
            self.session.append_history_line()
            self.settings.save()
        finally:
            self.destroy()

    # ================================================================== loop de atualização (thread principal)
    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.engine.events.get_nowait()
                if kind == "state":
                    self._set_state(payload)
                    if payload == "running":
                        self._set_status("Detecção em andamento.", "ok")
                    elif payload == "paused":
                        self._set_status("Pausado.", "warn")
                elif kind == "status":
                    level, text = payload
                    self._set_status(text, level)
                    if level == "error":
                        messagebox.showerror("Câmera", text, parent=self)
                elif kind == "model":
                    self._model_info = payload
                    name = Path(payload.weights).name
                    self.model_chip.configure(text=f"MODELO {name}" + (" · DEMO" if payload.is_demo else ""),
                                              text_color=T.WARN if payload.is_demo else T.OK)
                    self.device_chip.configure(text=f"DISPOSITIVO {payload.device.upper()}", text_color=T.ACCENT)
                elif kind == "error":
                    self._set_status(str(payload), "error")
                    messagebox.showerror("Erro", str(payload), parent=self)
                elif kind == "stopped":
                    self.engine.stop()
        except queue.Empty:
            pass

    def _poll(self) -> None:
        self._drain_events()
        res = self.engine.latest_result()
        if res is not None and res.frame_id != self._last_frame_id and self._state != "idle":
            self._last_frame_id, self._last_result = res.frame_id, res
            self._render()
        now = time.monotonic()
        if now - self._last_stats >= self.settings.ui.stats_refresh_ms / 1000.0:
            self._last_stats = now
            self._refresh_stats()
            text = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
            if text != self._clock_text:
                self._clock_text = text
                self.clock.configure(text=text)
        self.after(15, self._poll)

    def _refresh_stats(self) -> None:
        self.stats.update_stats(self.session.snapshot())
        for seq, ev in self.session.events_since(self._hist_seq):
            self.history.add(ev)
            self._hist_seq = seq

    # ================================================================== vídeo
    def _on_canvas_resize(self, ev) -> None:
        self.canvas.coords(self._placeholder, ev.width // 2, ev.height // 2)
        if self._last_result is not None and self._state != "idle":
            self._render()

    def _clear_video(self) -> None:
        self._last_result, self._last_frame_id, self._photo = None, -1, None
        if self._img_item is not None:
            self.canvas.delete(self._img_item)
            self._img_item = None
        self.canvas.itemconfigure(self._placeholder, state="normal")

    def _render(self) -> None:
        res = self._last_result
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if res is None or cw < 30 or ch < 30:
            return
        h, w = res.frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(res.frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        overlay.draw_detections(img, res.detections, scale, self.materials)
        info = self._model_info
        overlay.draw_hud(img, self._state, res.capture_fps, res.infer_fps, res.infer_ms,
                         info.device if info else "—", bool(info and info.is_demo))
        self._photo = ImageTk.PhotoImage(img)   # guarda a referência (senão o GC apaga a imagem)
        x, y = (cw - nw) // 2, (ch - nh) // 2
        if self._img_item is None:
            self._img_item = self.canvas.create_image(x, y, anchor="nw", image=self._photo)
            self.canvas.itemconfigure(self._placeholder, state="hidden")
        else:
            self.canvas.itemconfigure(self._img_item, image=self._photo)
            self.canvas.coords(self._img_item, x, y)


def run() -> None:
    settings = Settings.load()
    materials = MaterialsCfg.load()
    MainWindow(settings, materials).mainloop()
