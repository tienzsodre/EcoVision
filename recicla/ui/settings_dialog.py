"""Tela de configurações (câmera, modelo, rastreamento e materiais/preços)."""
from __future__ import annotations

import copy
from tkinter import filedialog

import customtkinter as ctk

from ..config import MaterialsCfg, Settings, parse_source
from . import theme as T

DEVICES = ["auto", "mps", "cpu", "cuda"]
IMGSZ = ["320", "416", "512", "640", "768"]


class SettingsDialog(ctk.CTkToplevel):
    """`on_save(old_settings, new_settings)` é chamado após validar e gravar."""

    def __init__(self, parent, settings: Settings, materials: MaterialsCfg, fonts: dict, on_save):
        super().__init__(parent)
        self.title("Configurações do sistema")
        self.geometry("620x560")
        self.configure(fg_color=T.BG)
        self.transient(parent)
        self.settings, self.materials, self.f, self.on_save = settings, materials, fonts, on_save
        self.vars: dict[str, ctk.Variable] = {}
        self._mat_vars: dict[str, tuple[ctk.StringVar, ctk.StringVar]] = {}

        tabs = ctk.CTkTabview(self, fg_color=T.PANEL, segmented_button_selected_color="#0e7490",
                              segmented_button_selected_hover_color="#0e7490", segmented_button_unselected_color=T.PANEL_ALT,
                              segmented_button_unselected_hover_color=T.BORDER, text_color=T.TEXT)
        tabs.pack(fill="both", expand=True, padx=14, pady=(14, 6))
        for name in ("Câmera", "Modelo", "Rastreamento", "Materiais"):
            tabs.add(name)
        self._tab_camera(tabs.tab("Câmera"))
        self._tab_model(tabs.tab("Modelo"))
        self._tab_tracking(tabs.tab("Rastreamento"))
        self._tab_materials(tabs.tab("Materiais"))

        self.error = ctk.CTkLabel(self, text="", text_color=T.DANGER, font=fonts["small"])
        self.error.pack(anchor="w", padx=18)
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(0, 14))
        ctk.CTkButton(bar, text="Salvar", width=120, fg_color=T.ACCENT, text_color="#06222a", hover_color="#67e8f9",
                      command=self._save).pack(side="right")
        ctk.CTkButton(bar, text="Cancelar", width=100, fg_color=T.PANEL_ALT, hover_color=T.BORDER,
                      command=self.destroy).pack(side="right", padx=8)
        self.after(150, self.focus_force)

    # -------------------------------------------------------------- helpers
    def _row(self, parent, r: int, label: str, key: str, value, width: int = 110, hint: str = ""):
        ctk.CTkLabel(parent, text=label, anchor="w", font=self.f["body"]).grid(row=r, column=0, sticky="w", padx=8, pady=6)
        var = ctk.StringVar(value=str(value))
        self.vars[key] = var
        ctk.CTkEntry(parent, textvariable=var, width=width).grid(row=r, column=1, sticky="w", pady=6)
        if hint:
            ctk.CTkLabel(parent, text=hint, text_color=T.MUTED, font=self.f["tiny"], anchor="w").grid(row=r, column=2, sticky="w", padx=8)
        return var

    def _tab_camera(self, tab):
        c = self.settings.camera
        self._row(tab, 0, "Fonte (índice ou vídeo)", "cam.source", c.source, width=230,
                  hint="0, 1, 2… ou caminho de um vídeo")
        ctk.CTkButton(tab, text="Escolher vídeo…", width=130, fg_color=T.PANEL_ALT, hover_color=T.BORDER,
                      command=self._pick_video).grid(row=1, column=1, sticky="w", pady=2)
        self._row(tab, 2, "Largura (px)", "cam.width", c.width)
        self._row(tab, 3, "Altura (px)", "cam.height", c.height)
        self._row(tab, 4, "FPS da câmera", "cam.fps", c.fps)
        self.vars["cam.mirror"] = ctk.BooleanVar(value=c.mirror)
        ctk.CTkSwitch(tab, text="Espelhar imagem", variable=self.vars["cam.mirror"]).grid(row=5, column=0, columnspan=2, sticky="w", padx=8, pady=10)
        ctk.CTkLabel(tab, text="Mudanças de câmera valem ao iniciar a câmera novamente.", text_color=T.MUTED,
                     font=self.f["tiny"]).grid(row=6, column=0, columnspan=3, sticky="w", padx=8)

    def _tab_model(self, tab):
        m = self.settings.model
        self._row(tab, 0, "Pesos do modelo (.pt)", "mod.path", m.path, width=290)
        ctk.CTkButton(tab, text="Procurar…", width=100, fg_color=T.PANEL_ALT, hover_color=T.BORDER,
                      command=self._pick_weights).grid(row=0, column=2, padx=6)
        ctk.CTkLabel(tab, text="Dispositivo", anchor="w", font=self.f["body"]).grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.vars["mod.device"] = ctk.StringVar(value=m.device)
        ctk.CTkOptionMenu(tab, values=DEVICES, variable=self.vars["mod.device"], width=110).grid(row=1, column=1, sticky="w")
        ctk.CTkLabel(tab, text="Resolução da IA (imgsz)", anchor="w", font=self.f["body"]).grid(row=2, column=0, sticky="w", padx=8, pady=6)
        self.vars["mod.imgsz"] = ctk.StringVar(value=str(m.imgsz))
        ctk.CTkOptionMenu(tab, values=IMGSZ, variable=self.vars["mod.imgsz"], width=110).grid(row=2, column=1, sticky="w")
        ctk.CTkLabel(tab, text="menor = mais FPS, maior = mais precisão", text_color=T.MUTED, font=self.f["tiny"]).grid(row=2, column=2, sticky="w", padx=8)
        self._row(tab, 3, "IoU (NMS)", "mod.iou", m.iou)
        self._row(tab, 4, "Máx. detecções", "mod.max_det", m.max_det)
        self._row(tab, 5, "Teto de FPS da IA", "mod.max_infer_fps", m.max_infer_fps)
        ctk.CTkLabel(tab, text="A confiança mínima fica na tela principal (deslizante).\nPesos/dispositivo valem ao reiniciar.",
                     text_color=T.MUTED, font=self.f["tiny"], justify="left").grid(row=6, column=0, columnspan=3, sticky="w", padx=8, pady=8)

    def _tab_tracking(self, tab):
        t = self.settings.tracking
        self._row(tab, 0, "Quadros p/ confirmar", "trk.min_hits", t.min_hits, hint="maior = menos falsos positivos")
        self._row(tab, 1, "Janela de reidentificação (s)", "trk.reid_window_s", t.reid_window_s, hint="0 desliga")
        self._row(tab, 2, "IoU de reidentificação", "trk.reid_iou", t.reid_iou, hint="0 a 1")
        ctk.CTkLabel(tab, text=("Cada objeto é contado uma vez, quando seu ID de rastreamento é confirmado.\n"
                                "A reidentificação evita contar de novo um objeto que o rastreador perdeu\n"
                                "por instantes e devolveu com outro ID."), text_color=T.MUTED, font=self.f["tiny"],
                     justify="left").grid(row=3, column=0, columnspan=3, sticky="w", padx=8, pady=10)

    def _tab_materials(self, tab):
        ctk.CTkLabel(tab, text="Valores ESTIMADOS: contagem × peso médio × preço/kg. Ajuste à sua realidade.",
                     text_color=T.WARN, font=self.f["small"]).grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))
        for c, t in enumerate(("Material", "Peso médio (g)", f"Preço ({self.materials.currency}/kg)")):
            ctk.CTkLabel(tab, text=t, text_color=T.MUTED, font=self.f["tiny_b"]).grid(row=1, column=c, sticky="w", padx=8)
        for i, m in enumerate(self.materials, start=2):
            ctk.CTkLabel(tab, text=m.label, anchor="w", font=self.f["body"]).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            w, p = ctk.StringVar(value=f"{m.avg_weight_g:g}"), ctk.StringVar(value=f"{m.price_per_kg:g}")
            self._mat_vars[m.id] = (w, p)
            ctk.CTkEntry(tab, textvariable=w, width=90).grid(row=i, column=1, sticky="w", pady=3, padx=8)
            ctk.CTkEntry(tab, textvariable=p, width=90).grid(row=i, column=2, sticky="w", pady=3, padx=8)

    def _pick_video(self):
        path = filedialog.askopenfilename(parent=self, title="Escolher vídeo de demonstração",
                                          filetypes=[("Vídeos", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("Todos", "*.*")])
        if path:
            self.vars["cam.source"].set(path)

    def _pick_weights(self):
        path = filedialog.askopenfilename(parent=self, title="Escolher pesos YOLO", filetypes=[("Pesos", "*.pt *.onnx"), ("Todos", "*.*")])
        if path:
            self.vars["mod.path"].set(path)

    # -------------------------------------------------------------- salvar
    def _save(self):
        v = {k: var.get() for k, var in self.vars.items()}
        old = copy.deepcopy(self.settings)
        try:
            s = self.settings
            s.camera.source = parse_source(v["cam.source"])
            s.camera.width, s.camera.height, s.camera.fps = int(v["cam.width"]), int(v["cam.height"]), int(v["cam.fps"])
            s.camera.mirror = bool(v["cam.mirror"])
            s.model.path, s.model.device, s.model.imgsz = v["mod.path"].strip(), v["mod.device"], int(v["mod.imgsz"])
            s.model.iou, s.model.max_det = float(v["mod.iou"]), int(v["mod.max_det"])
            s.model.max_infer_fps = int(v["mod.max_infer_fps"])
            s.tracking.min_hits = max(1, int(v["trk.min_hits"]))
            s.tracking.reid_window_s = max(0.0, float(v["trk.reid_window_s"].replace(",", ".")))
            s.tracking.reid_iou = min(1.0, max(0.0, float(v["trk.reid_iou"].replace(",", "."))))
            for mid, (w, p) in self._mat_vars.items():
                m = self.materials.get(mid)
                m.avg_weight_g = max(0.0, float(w.get().replace(",", ".")))
                m.price_per_kg = max(0.0, float(p.get().replace(",", ".")))
        except ValueError:
            self.settings.__dict__.update(old.__dict__)  # desfaz atribuições parciais
            self.error.configure(text="Valor inválido: confira os campos numéricos.")
            return
        s.save()
        self.materials.save()
        self.on_save(old, s)
        self.destroy()
