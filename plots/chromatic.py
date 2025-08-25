#!/usr/bin/env python3
# batch_most_diff_color_vs_anchor.py
import os
from collections import Counter
from PIL import Image, ImageDraw
import numpy as np


# ========= CONFIG: edit these =========
ANCHOR_IMG = "CEM_RESULTS_3/dinowm_agent_color/env_10/goal_10.png"
ROOT_DIR   = "CEM_RESULTS_3/dinowm_agent_color"
TOP_K = 5
MAX_SIDE = 700
WHITE_THRESH = 235
BLACK_THRESH = 20
QUANT = 6
MATCH_TOL = 12.0               # RGB Euclidean distance to treat as “same”
BLOCK_SIZE = 64                # square size in pixels
GAP = 0                        # gap between squares (px); keep 0 for flush
OUT_IMAGE = "palette_row.png"
# =================

def load_png(path):
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / float(max(w, h))
        im = im.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return im

def extract_pixels(im):
    arr = np.array(im, dtype=np.uint8)
    rgb = arr[..., :3]
    a   = arr[..., 3]
    vis = a > 0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    near_white = (r >= WHITE_THRESH) & (g >= WHITE_THRESH) & (b >= WHITE_THRESH)
    near_black = (r <= BLACK_THRESH) & (g <= BLACK_THRESH) & (b <= BLACK_THRESH)
    keep = vis & ~(near_white | near_black)
    return rgb[keep]

def topk_colors(rgb_arr, k=5, quant=QUANT):
    if rgb_arr.size == 0:
        return []
    q = (rgb_arr // quant) * quant
    counts = Counter(map(tuple, q.tolist()))
    total = sum(counts.values())
    return [(tuple(c), n, n/total) for c, n in counts.most_common(k)]

def euclid_rgb(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    return float(np.linalg.norm(a - b))

def non_matching(B_top, A_top, tol=MATCH_TOL):
    A_colors = [c for c,_,_ in A_top]
    out = []
    for cB, nB, fB in B_top:
        d = min((euclid_rgb(cB, cA) for cA in A_colors), default=1e9)
        if d > tol:
            out.append({"rgb": cB, "dist": d, "frac": fB})
    return out

def pick_most_diff(nonmatches):
    if not nonmatches: return None
    return max(nonmatches, key=lambda d: (d["dist"], d["frac"]))

def find_goal_pngs(root):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            low = fn.lower()
            if low.startswith("goal") and low.endswith(".png"):
                yield os.path.join(dp, fn)

def make_palette_row(colors, block=BLOCK_SIZE, gap=GAP, out=OUT_IMAGE):
    n = len(colors)
    w = n * block + (n - 1) * gap
    h = block
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = 0
    for rgb in colors:
        draw.rectangle([x, 0, x + block - 1, h - 1], fill=rgb)
        x += block + gap
    img.save(out)
    print(f"Saved palette: {out}")

if __name__ == "__main__":
    # Anchor color set
    anchor_im = load_png(ANCHOR_IMG)
    anchor_px = extract_pixels(anchor_im)
    anchor_top = topk_colors(anchor_px, k=TOP_K)
    if not anchor_top:
        raise SystemExit("Anchor produced no colors; adjust thresholds.")
    anchor_rgb = anchor_top[0][0]

    colors = [anchor_rgb]  # first square = anchor

    # For each goal*.png, compute the most-different color vs anchor and append
    for path in sorted(find_goal_pngs(ROOT_DIR)):
        im = load_png(path)
        px = extract_pixels(im)
        topB = topk_colors(px, k=TOP_K)
        nm = non_matching(topB, anchor_top, tol=MATCH_TOL)
        chosen = pick_most_diff(nm)
        if chosen:
            colors.append(chosen["rgb"])
        # If no non-match, skip silently (no “diff stuff”)

    if len(colors) == 1:
        raise SystemExit("No goal images produced a non-matching color.")

    make_palette_row(colors, out=OUT_IMAGE)
