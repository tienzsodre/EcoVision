"""Rastreador falso (sem PyTorch) para testar câmera + motor + contagem + interface."""
import cv2
import numpy as np

from recicla.core.detector import ModelInfo
from recicla.core.types import RawTrack


class FakeTracker:
    def __init__(self):
        self.conf, self.iou, self.n = 0.4, 0.5, 0
        self.resets = 0

    def load(self, warmup_shape=(720, 1280)):
        return ModelInfo("fake.pt", "cpu", False, ["garrafa_pet", "lata_aluminio"],
                         ["garrafa_pet", "lata_aluminio"], [])

    def reset_tracker(self):
        self.resets += 1

    def infer(self, frame):
        self.n += 1
        h, w = frame.shape[:2]
        x = 20 + (self.n * 3) % (w // 2)
        out = [RawTrack(1, "garrafa_pet", 0.9, (x, h * 0.3, x + w * 0.15, h * 0.7))]
        if self.n > 10:
            out.append(RawTrack(2, "lata_aluminio", 0.8, (w * 0.7, h * 0.5, w * 0.85, h * 0.8)))
        if self.n > 25:
            out.append(RawTrack(None, "papel", 0.6, (5, 5, 60, 60)))   # sem ID: nunca conta
        return out


def make_video(path, frames=90, size=(320, 240), fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), (60, 80, 40), np.uint8)
        cv2.circle(img, (30 + i * 3 % 250, 120), 25, (200, 200, 220), -1)
        vw.write(img)
    vw.release()
    return path
