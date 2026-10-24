"""Draws the Depth Studio app icon: layered hills coloured far→near with the same
turbo map the app uses, on a dark macOS-grid rounded square. Writes an .iconset."""
import os
import sys

import cv2
import matplotlib
import numpy as np

S = 1024
SS = 4                      # supersample for smooth edges
N = S * SS
turbo = matplotlib.colormaps["turbo"]


def rgb(t):
    r, g, b, _ = turbo(t)
    return (int(b * 255), int(g * 255), int(r * 255))   # BGR for cv2


img = np.zeros((N, N, 3), np.uint8)
img[:] = (46, 22, 18)       # deep navy: "farther than everything"
x = np.linspace(0, 1, 400)
# far → near: lower, bigger, warmer
layers = [(0.50, 0.07, 1.3, 0.10), (0.60, 0.08, 1.9, 0.35), (0.71, 0.08, 1.5, 0.62), (0.83, 0.07, 2.4, 0.92)]
for base, amp, freq, t in layers:
    y = base - amp * (0.6 * np.sin(2 * np.pi * freq * x + t * 7) + 0.4 * np.sin(2 * np.pi * freq * 2.3 * x + t * 3))
    pts = np.stack([x * N, y * N], 1).astype(np.int32)
    poly = np.concatenate([pts, [[N, N], [0, N]]]).astype(np.int32)
    cv2.fillPoly(img, [poly], rgb(t), lineType=cv2.LINE_AA)

# Apple icon grid: 824 px body centred on 1024, corner radius ~185.
mask = np.zeros((N, N), np.uint8)
m, r = 100 * SS, 185 * SS
cv2.rectangle(mask, (m + r, m), (N - m - r, N - m), 255, -1)
cv2.rectangle(mask, (m, m + r), (N - m, N - m - r), 255, -1)
for cx, cy in [(m + r, m + r), (N - m - r, m + r), (m + r, N - m - r), (N - m - r, N - m - r)]:
    cv2.circle(mask, (cx, cy), r, 255, -1, lineType=cv2.LINE_AA)

big = np.dstack([img, mask])
icon = cv2.resize(big, (S, S), interpolation=cv2.INTER_AREA)

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for size in (16, 32, 128, 256, 512):
    for scale in (1, 2):
        px = size * scale
        name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
        cv2.imwrite(os.path.join(out, name), cv2.resize(icon, (px, px), interpolation=cv2.INTER_AREA))
