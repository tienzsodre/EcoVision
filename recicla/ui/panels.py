"""Painéis da interface: estatísticas e monitoramento (histórico)."""
from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import ttk

import customtkinter as ctk

from ..config import MaterialsCfg
from ..core.estimates import fmt_money, fmt_weight, item_estimate
from ..core.session import Snapshot
from ..core.types import CountEvent
from . import theme as T


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def card(parent, **kw) -> ctk.CTkFrame:
    return ctk.CTkFrame(parent, fg_color=T.PANEL, corner_radius=T.RADIUS, border_width=1,
                        border_color=T.BORDER, **kw)


def chip(parent, text: str, fg: str = T.PANEL_ALT, color: str = T.MUTED, font=None) -> ctk.CTkLabel:
    return ctk.CTkLabel(parent, text=text, fg_color=fg, text_color=color, corner_radius=8,
                        padx=10, pady=3, font=font)


class StatsPanel(ctk.CTkFrame):
    """Total, peso e valor ESTIMADOS e uma linha por categoria (com barra de participação)."""

    def __init__(self, parent, materials: MaterialsCfg, fonts: dict):
        super().__init__(parent, fg_color=T.PANEL, corner_radius=T.RADIUS, border_width=1, border_color=T.BORDER)
        self.materials, self.f = materials, fonts
        self._cache: dict[str, str] = {}
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=T.PAD + 2, pady=(T.PAD, 0))
        ctk.CTkLabel(head, text="ESTATÍSTICAS DA SESSÃO", font=fonts["small_b"], text_color=T.MUTED).pack(side="left")
        chip(head, "ESTIMATIVA", fg="#3b2a0a", color=T.WARN, font=fonts["tiny_b"]).pack(side="right")

        self.total_lbl = ctk.CTkLabel(self, text="0", font=fonts["huge"], text_color=T.TEXT, anchor="w")
        self.total_lbl.grid(row=1, column=0, sticky="w", padx=T.PAD + 2, pady=(0, 0))
        side = ctk.CTkFrame(self, fg_color="transparent")
        side.grid(row=1, column=1, sticky="e", padx=T.PAD + 2)
        self.time_lbl = ctk.CTkLabel(side, text="Duração 00:00:00", font=fonts["small"], text_color=T.MUTED, anchor="e")
        self.time_lbl.pack(anchor="e")
        self.rate_lbl = ctk.CTkLabel(side, text="0 itens/min (último min)", font=fonts["small"], text_color=T.MUTED, anchor="e")
        self.rate_lbl.pack(anchor="e")
        ctk.CTkLabel(self, text="OBJETOS IDENTIFICADOS (contados 1x por ID)", font=fonts["tiny"], text_color=T.MUTED,
                     anchor="w").grid(row=2, column=0, columnspan=2, sticky="w", padx=T.PAD + 2)

        self.weight_lbl = self._tile(3, 0, "PESO TOTAL (est.)")
        self.value_lbl = self._tile(3, 1, "VALOR TOTAL (est.)")

        self.rows_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", height=292)
        self.rows_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=(T.PAD - 4, T.PAD - 4), pady=(0, T.PAD - 4))
        self.rowconfigure(4, weight=1)
        self._rows: dict[str, dict] = {}
        self._build_rows()

    def _tile(self, r: int, c: int, title: str) -> ctk.CTkLabel:
        tile = ctk.CTkFrame(self, fg_color=T.PANEL_ALT, corner_radius=10)
        tile.grid(row=r, column=c, sticky="ew", padx=(T.PAD + 2 if c == 0 else 4, 4 if c == 0 else T.PAD + 2), pady=T.PAD - 2)
        ctk.CTkLabel(tile, text=title, font=self.f["tiny"], text_color=T.MUTED).pack(anchor="w", padx=10, pady=(6, 0))
        lbl = ctk.CTkLabel(tile, text="—", font=self.f["big"], text_color=T.TEXT)
        lbl.pack(anchor="w", padx=10, pady=(0, 6))
        return lbl

    def _build_rows(self) -> None:
        for w in self.rows_frame.winfo_children():
            w.destroy()
        self._rows.clear()
        self.rows_frame.columnconfigure(0, weight=1)
        for i, m in enumerate(self.materials):
            fr = ctk.CTkFrame(self.rows_frame, fg_color="transparent")
            fr.grid(row=i, column=0, sticky="ew", pady=1, padx=4)
            fr.columnconfigure(1, weight=1)
            ctk.CTkFrame(fr, width=10, height=10, corner_radius=5, fg_color=m.color).grid(row=0, column=0, padx=(2, 8))
            ctk.CTkLabel(fr, text=m.label, font=self.f["body"], text_color=T.TEXT, anchor="w").grid(row=0, column=1, sticky="w")
            cnt = ctk.CTkLabel(fr, text="0", font=self.f["body_b"], text_color=T.TEXT, anchor="e")
            cnt.grid(row=0, column=2, sticky="e")
            bar = ctk.CTkProgressBar(fr, height=4, progress_color=m.color, fg_color=T.PANEL_ALT, corner_radius=3)
            bar.set(0)
            bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 0))
            sub = ctk.CTkLabel(fr, text="0 g · R$ 0,00", font=self.f["tiny"], text_color=T.MUTED, anchor="e")
            sub.grid(row=1, column=2, sticky="e", padx=(8, 0))
            self._rows[m.id] = {"cnt": cnt, "bar": bar, "sub": sub}

    def _set(self, key: str, widget, text: str) -> None:
        if self._cache.get(key) != text:       # só mexe no Tk se o valor mudou
            self._cache[key] = text
            widget.configure(text=text)

    def update_stats(self, s: Snapshot) -> None:
        cur = self.materials.currency
        self._set("total", self.total_lbl, str(s.total))
        self._set("time", self.time_lbl, f"Duração {fmt_duration(s.elapsed_s)}")
        self._set("rate", self.rate_lbl, f"{s.rate_last_min} itens/min (último min)")
        self._set("weight", self.weight_lbl, "≈ " + fmt_weight(s.total_weight_kg))
        self._set("value", self.value_lbl, "≈ " + fmt_money(s.total_value, cur))
        biggest = max((c.count for c in s.by_material.values()), default=0) or 1
        for mid, row in self._rows.items():
            st = s.by_material[mid]
            self._set(f"c{mid}", row["cnt"], str(st.count))
            self._set(f"s{mid}", row["sub"], f"≈ {fmt_weight(st.weight_kg)} · {fmt_money(st.value, cur)}")
            share = round(st.count / biggest, 2)
            if self._cache.get(f"b{mid}") != share:
                self._cache[f"b{mid}"] = share
                row["bar"].set(share)

    def refresh_materials(self, materials: MaterialsCfg) -> None:
        self.materials = materials
        self._cache.clear()
        self._build_rows()


class HistoryPanel(ctk.CTkFrame):
    COLS = (("hora", "Hora", 72), ("id", "ID", 52), ("cat", "Categoria", 112),
            ("peso", "Peso*", 62), ("valor", "Valor*", 74))

    def __init__(self, parent, materials: MaterialsCfg, fonts: dict, max_rows: int = 200):
        super().__init__(parent, fg_color=T.PANEL, corner_radius=T.RADIUS, border_width=1, border_color=T.BORDER)
        self.materials, self.max_rows = materials, max_rows
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=T.PAD + 2, pady=(T.PAD, 4))
        ctk.CTkLabel(head, text="MONITORAMENTO · ÚLTIMAS DETECÇÕES", font=fonts["small_b"], text_color=T.MUTED).pack(side="left")
        ctk.CTkLabel(head, text="* estimativa", font=fonts["tiny"], text_color=T.WARN).pack(side="right")

        style = ttk.Style(self)
        style.theme_use("clam")  # o tema nativo do macOS ignora cores customizadas
        fam = fonts["family"]
        style.configure("Rec.Treeview", background=T.PANEL_ALT, fieldbackground=T.PANEL_ALT, foreground=T.TEXT,
                        rowheight=25, borderwidth=0, font=(fam, 11))
        style.configure("Rec.Treeview.Heading", background=T.PANEL, foreground=T.MUTED, relief="flat",
                        font=(fam, 10, "bold"), borderwidth=0)
        style.map("Rec.Treeview", background=[("selected", "#1f3a52")], foreground=[("selected", T.TEXT)])
        style.map("Rec.Treeview.Heading", background=[("active", T.PANEL)])

        wrap = tk.Frame(self, bg=T.PANEL)
        wrap.pack(fill="both", expand=True, padx=T.PAD - 2, pady=(0, T.PAD - 2))
        self.tree = ttk.Treeview(wrap, columns=[c[0] for c in self.COLS], show="headings", style="Rec.Treeview",
                                 selectmode="browse")
        for key, title, width in self.COLS:
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="w" if key == "cat" else "center", stretch=(key == "cat"))
        sb = ctk.CTkScrollbar(wrap, command=self.tree.yview, fg_color=T.PANEL)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._tag_colors()

    def _tag_colors(self) -> None:
        for m in self.materials:
            self.tree.tag_configure(m.id, foreground=m.color)

    def add(self, ev: CountEvent) -> None:
        m = self.materials.get(ev.material)
        kg, val = item_estimate(m) if m else (0.0, 0.0)
        self.tree.insert("", 0, tags=(ev.material,), values=(
            datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S"), f"#{ev.track_id}",
            m.label if m else ev.material,
            f"{kg * 1000:.0f} g", fmt_money(val, self.materials.currency)))
        children = self.tree.get_children()
        for extra in children[self.max_rows:]:
            self.tree.delete(extra)

    def clear(self) -> None:
        self.tree.delete(*self.tree.get_children())

    def refresh_materials(self, materials: MaterialsCfg) -> None:
        self.materials = materials
        self._tag_colors()
