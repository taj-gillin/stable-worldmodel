import re
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Folder names: dinowm_<category>_<perturbation>
NAME_RX = re.compile(r"^dinowm_([^_]+)_(.+)$")  # <category>, <perturbation>

def _find_goal_in_folder(folder: Path):
    """
    Try common goal filename patterns in a folder and return RGB image or None.
    """
    candidates = []
    candidates += sorted(folder.glob("goal_*.png"))
    candidates += [folder / "goal.png"]
    candidates += sorted(folder.glob("goal*.jpg")) + sorted(folder.glob("goal*.jpeg"))
    for p in candidates:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None

def _pick_random_goal_from_perturbation(perturbation_dir: Path, rng: random.Random):
    """
    Each perturbation has N subfolders; choose one at random and read its goal image.
    Returns (chosen_subfolder_name, img) or (None, None) if none found.
    """
    # Consider only subdirectories
    subdirs = [d for d in sorted(perturbation_dir.iterdir()) if d.is_dir()]
    if not subdirs:
        # In case the goals are directly inside the perturbation folder (fallback)
        img = _find_goal_in_folder(perturbation_dir)
        return (perturbation_dir.name, img) if img is not None else (None, None)

    rng.shuffle(subdirs)  # randomize order, then pick the first that has a goal
    for sd in subdirs:
        img = _find_goal_in_folder(sd)
        if img is not None:
            return (sd.name, img)
    return (None, None)

def _nice_sort_key(s: str):
    m = re.match(r"(\D*)(\d+)$", s)
    if m:
        head, num = m.groups()
        return (head, int(num))
    return (s, 0)

def make_category_row_pdfs(
    root_folder,
    out_dir="plots/res",
    dpi=150,
    label_fontsize=16,
    max_height_px=None,
    seed=234564,  # set for reproducible random choices across runs
    include_chosen_subdir_in_label=False,
):
    """
    Root contains subfolders named 'dinowm_<category>_<perturbation>'.
    For each <category>, create one PDF (<category>.pdf) with a single row:
    one image per perturbation, where each image is chosen randomly from
    among that perturbation's N subfolders (each containing a goal).

    Options:
      - max_height_px: downscale each chosen image to this height (keep aspect)
      - seed: RNG seed for reproducible picks
      - include_chosen_subdir_in_label: show 'perturbation (picked_subdir)' in title
    """
    rng = random.Random(seed)

    root = Path(root_folder)
    if not root.exists():
        raise FileNotFoundError(root)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by category
    groups = {}  # category -> list[(perturbation, full_path)]
    for p in sorted([d for d in root.iterdir() if d.is_dir()]):
        m = NAME_RX.match(p.name)
        if not m:
            continue
        category, perturbation = m.groups()
        groups.setdefault(category, []).append((perturbation, p))

    if not groups:
        raise ValueError("No folders matching 'dinowm_<category>_<perturbation>' found.")

    for category, items in groups.items():
        items = sorted(items, key=lambda t: _nice_sort_key(t[0]))

        records = []  # (label, img)
        for perturbation, pert_dir in items:
            chosen_subdir, img = _pick_random_goal_from_perturbation(pert_dir, rng)
            if img is None:
                print(f"⚠️  Skipping (no goal found in any subfolder): {pert_dir}")
                continue
            if max_height_px and img.shape[0] > max_height_px:
                h, w = img.shape[:2]
                scale = max_height_px / h
                img = cv2.resize(img, (max(1, int(round(w * scale))), max_height_px), interpolation=cv2.INTER_AREA)

            label = perturbation
            if include_chosen_subdir_in_label and chosen_subdir:
                label = f"{perturbation} ({chosen_subdir})"
            records.append((label, img))

        if not records:
            print(f"⚠️  No images for category '{category}'. Skipping.")
            continue

        # Compute total canvas size (side-by-side)
        H = max(im.shape[0] for _, im in records)
        W = sum(im.shape[1] for _, im in records)
        fig_w, fig_h = W / dpi, H / dpi

        fig = plt.figure(figsize=(fig_w, fig_h))

        fig, axes = plt.subplots(1, len(records), figsize=(W/dpi, H/dpi), dpi=dpi)
        for ax, (label, img) in zip(axes, records):
            ax.imshow(img)
            ax.axis("off")
        fig.subplots_adjust(wspace=0.1)

        # x_off = 0
        # for label, img in records:
        #     h, w = img.shape[:2]
        #     left = x_off / W
        #     ax_w = w / W
        #     ax = fig.add_axes([left, 0, ax_w, 1.0])
        #     ax.imshow(img)
        #     ax.axis("off")
        #     x_off += w

        out_path = out_dir / f"{category}.pdf"
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"✅ Wrote {out_path}")

# Example
if __name__ == "__main__":
    make_category_row_pdfs(
        root_folder="CEM_RESULTS",
        out_dir="plots/res",
        dpi=150,
        label_fontsize=16,
        max_height_px=512,         # or None for full res
        seed=1234,                   # reproducible random picks
        include_chosen_subdir_in_label=True,
    )
