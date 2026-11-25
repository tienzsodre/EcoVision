"""
Desenho das caixas, rótulos e HUD sobre a imagem JÁ REDIMENSIONADA para a tela.

Desenhar aqui (e não no quadro original) deixa o texto nítido em qualquer tamanho de
janela, mantém as linhas com espessura constante e aceita acentos (cv2.putText não).
"""
from __future__ import annotations

import sys
from functools import lru_cache

from PIL import ImageDraw, ImageFont

from ..config import MaterialsCfg
from ..core.types import Detection

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


@lru_cache(maxsize=16)
def font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _text_size(draw: ImageDraw.ImageDraw, text: str, f) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    return r - l, b - t


def _luma(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def draw_detections(img, detections: list[Detection], scale: float, materials: MaterialsCfg) -> None:
    d = ImageDraw.Draw(img)
    f = font(13)
    W, H = img.size
    for det in detections:
        mat = materials.get(det.material)
        rgb = hex_to_rgb(mat.color) if mat else (200, 200, 200)
        x1, y1, x2, y2 = (int(v * scale) for v in det.box)
        counted = det.state == "counted"
        width = 3 if counted else 1
        d.rectangle([x1, y1, x2, y2], outline=rgb, width=width)
        if counted:  # cantos reforçados (estilo mira)
            c = max(8, min(x2 - x1, y2 - y1) // 5)
            for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                d.line([px, py, px + dx * c, py], fill=rgb, width=5)
                d.line([px, py, px, py + dy * c], fill=rgb, width=5)
        label = mat.label if mat else det.material
        ident = f"#{det.track_id} " if det.track_id is not None else ""
        confidence = f"{det.conf * 100:.0f}%" if det.conf is not None else ""
        text = f"{label}  {ident}{confidence}".rstrip()
        tw, th = _text_size(d, text, f)
        bx1, by2 = max(0, min(x1, W - tw - 12)), max(th + 10, y1)
        by1 = by2 - th - 10
        if counted:
            d.rectangle([bx1, by1, bx1 + tw + 12, by2], fill=rgb)
            d.text((bx1 + 6, by1 + 3), text, fill=(10, 14, 19) if _luma(rgb) > 140 else (255, 255, 255), font=f)
        else:
            d.rectangle([bx1, by1, bx1 + tw + 12, by2], fill=(17, 24, 33), outline=rgb)
            d.text((bx1 + 6, by1 + 3), text, fill=rgb, font=f)


def _pill(d, xy, text, color, f, dot=True):
    tw, th = _text_size(d, text, f)
    x, y = xy
    pad_l = 26 if dot else 12
    w, h = tw + pad_l + 12, th + 14
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=(10, 14, 19), outline=(35, 48, 68))
    if dot:
        d.ellipse([x + 10, y + h // 2 - 5, x + 20, y + h // 2 + 5], fill=color)
    d.text((x + pad_l, y + 6), text, fill=(232, 238, 245), font=f)
    return w, h


def draw_hud(img, status: str, capture_fps: float, infer_fps: float, infer_ms: float,
             device: str, demo: bool) -> None:
    d = ImageDraw.Draw(img)
    W, H = img.size
    f = font(14)
    color, text = {
        "running": ((34, 197, 94), "CÂMERA ATIVA"),
        "paused": ((245, 158, 11), "PAUSADO"),
        "loading": ((34, 211, 238), "CARREGANDO MODELO…"),
    }.get(status, ((239, 68, 68), "SEM SINAL"))
    _pill(d, (14, 14), text, color, f)
    stats = f"FPS {infer_fps:4.1f}   IA {infer_ms:3.0f} ms   {device.upper()}"
    tw, _ = _text_size(d, stats, f)
    _pill(d, (W - tw - 52, 14), stats, (34, 211, 238), f, dot=False)
    if demo:
        msg = "MODELO DEMO (COCO) · NÃO TREINADO PARA RECICLÁVEIS"
        tw, th = _text_size(d, msg, f)
        d.rounded_rectangle([14, H - th - 34, 14 + tw + 24, H - 14], radius=8, fill=(120, 53, 15), outline=(245, 158, 11))
        d.text((26, H - th - 27), msg, fill=(254, 243, 199), font=f)
