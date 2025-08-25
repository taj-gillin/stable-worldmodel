import re
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def _sequence_from_env_folder(env_folder: Path, step=1):
    """Return list of RGB images forming [init] + video (drop first) + [goal]."""
    m = re.match(r"env_(.+)", env_folder.name)
    x = m.group(1) if m else None

    init_path = env_folder / (f"init_{x}.png" if x else "init.png")
    goal_path = env_folder / (f"goal_{x}.png" if x else "goal.png")

    mp4s = list(env_folder.glob(f"env_{x}*.mp4")) if x else []
    if not mp4s:
        mp4s = list(env_folder.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 in {env_folder}")

    init_img = cv2.cvtColor(cv2.imread(str(init_path)), cv2.COLOR_BGR2RGB)
    goal_img = cv2.cvtColor(cv2.imread(str(goal_path)), cv2.COLOR_BGR2RGB)

    cap = cv2.VideoCapture(str(mp4s[0]))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames in video: {mp4s[0]}")

    video_frames = frames[1::step] if len(frames) > 1 else []  # drop first frame
    return [init_img] + video_frames + [goal_img]


def save_env_rollouts_grid(
    parent_folder,
    env_ids=None,
    out_path="rollouts_grid.png",
    step=1,
    dpi=150,
    fontsize=20,
):
    """
    Plot multiple env rollouts as a matrix:
      - parent_folder: path to dir containing env_* subfolders
      - each row = one env sequence [init] + video (drop first) + [goal]
      - only the FIRST row has titles 'start' (first col) and 'goal' (last col)
      - last column (goal images) outlined with gold rectangle
    """
    parent = Path(parent_folder)
    if not parent.exists():
        raise FileNotFoundError(parent)

    all_envs = sorted([p for p in parent.iterdir() if p.is_dir() and p.name.startswith("env_")],
                      key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_",1)[1].isdigit() else p.name)

    if env_ids is None:
        envs = all_envs
    else:
        if isinstance(env_ids, int):
            envs = all_envs[:env_ids]
        else:
            wanted = set(str(i) for i in env_ids)
            envs = [p for p in all_envs if p.name.split("_", 1)[1] in wanted]

    if not envs:
        raise ValueError("No env_* folders selected.")

    sequences = [_sequence_from_env_folder(ef, step=step) for ef in envs]

    lengths = [len(s) for s in sequences]
    min_len = min(lengths)
    sequences = [s[:min_len] for s in sequences]  # truncate to shortest

    n_rows, n_cols = len(sequences), len(sequences[0])

    fig_w = max(2.2 * n_cols, 3)
    fig_h = max(2.2 * n_rows, 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(sequences[r][c])
            ax.axis("off")
            if r == 0 and c == 0:
                ax.set_title("start", fontsize=fontsize)
            elif r == 0 and c == n_cols - 1:
                ax.set_title("goal", fontsize=fontsize)

            # Add gold border to last column
            if c == n_cols - 1:
                rect = patches.Rectangle(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    fill=False,
                    color="gold",
                    linewidth=10
                )
                ax.add_patch(rect)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

    return sequences

import random
from pathlib import Path

if __name__ == "__main__":
    root = Path("CEM_RESULTS_3")
    output_paths = Path("plots/res")
    output_paths.mkdir(parents=True, exist_ok=True)

    for exp_folder in root.iterdir():
        if not exp_folder.is_dir():
            continue  # skip non-folders

        try:
            env_ids = random.sample(range(10), 3)
            print(f"Processing {exp_folder} with envs {env_ids} ...")

            out_path = output_paths / f"rollouts_grid_{exp_folder.name}.pdf"

            save_env_rollouts_grid(
                exp_folder,
                env_ids=env_ids,
                out_path=out_path,
                step=5,
                dpi=150,
                fontsize=32,
            )
        except Exception as e:
            print(f"⚠️ Skipped {exp_folder}: {e}")