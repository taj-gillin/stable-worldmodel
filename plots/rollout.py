import re
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_env_sequence_plot(folder, out_path="frames_plot.png", step=1, dpi=150, fontsize=18):
    """
    From a folder containing init_x.png, goal_x.png, and env_x-*.mp4:
      - Build a sequence [init] + video frames (drop first frame) + [goal]
      - Plot frames in one row
      - Put 'start' title above the first, 'goal' above the last
      - Save the figure to out_path
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Infer 'x' from folder name like 'env_3' → '3'
    m = re.match(r"env_(.+)", folder.name)
    x = m.group(1) if m else None

    # Resolve file paths
    init_path = folder / (f"init_{x}.png" if x else "init.png")
    goal_path = folder / (f"goal_{x}.png" if x else "goal.png")

    # Find matching mp4 (prefer env_x*.mp4, else any .mp4 in folder)
    mp4_candidates = list(folder.glob(f"env_{x}*.mp4")) if x else []
    if not mp4_candidates:
        mp4_candidates = list(folder.glob("*.mp4"))
    if not mp4_candidates:
        raise FileNotFoundError("No .mp4 file found in folder.")
    mp4_path = mp4_candidates[0]

    # Read init/goal images
    if not init_path.exists():
        raise FileNotFoundError(f"Missing init image: {init_path}")
    if not goal_path.exists():
        raise FileNotFoundError(f"Missing goal image: {goal_path}")

    init_img = cv2.cvtColor(cv2.imread(str(init_path)), cv2.COLOR_BGR2RGB)
    goal_img = cv2.cvtColor(cv2.imread(str(goal_path)), cv2.COLOR_BGR2RGB)

    # Read video frames
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {mp4_path}")

    # Drop the first frame, then apply step
    video_frames = frames[1::step] if len(frames) > 1 else []

    # Build full sequence
    seq = [init_img] + video_frames + [goal_img]

    # Plot and save
    fig_w = max(2.5 * len(seq), 3)  # adjust width scaling
    plt.figure(figsize=(fig_w, 3))
    for i, img in enumerate(seq):
        ax = plt.subplot(1, len(seq), i + 1)
        ax.imshow(img)
        ax.axis("off")
        if i == 0:
            ax.set_title("start", fontsize=fontsize)
        elif i == len(seq) - 1:
            ax.set_title("goal", fontsize=fontsize)
            # Add gold border rectangle
            rect = patches.Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                fill=False, color="gold", linewidth=6
            )
            ax.add_patch(rect)

    # Reduce spacing between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0)

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return np.array(seq, dtype=object)


if __name__ == "__main__":
    save_env_sequence_plot(
        "CEM_RESULTS/dinowm_bg_imagenet/env_3/",
        out_path="env_3_sequence.png",
        step=5,
        dpi=150,
        fontsize=20
    )
