"""
PinPadImage CLIP/SigLIP Heatmap Visualization.

For each (x,y) position on the PinPadImage grid, generates the state with the red
dot (laser pointer) at that location on the background image, computes the image
embedding, then measures L2 distance to goal text prompts. Produces heatmaps.

Usage:
    python scripts/visualization/pinpad_heatmap.py --config configs/pinpad_heatmap.yaml

    # Or with SLURM:
    sbatch slurm/pinpad_heatmap.slurm
"""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

import stable_worldmodel as swm
from stable_worldmodel.envs.pinpad.constants import X_BOUND, Y_BOUND, RENDER_SCALE


def load_embedding_model(model_name: str, device: str = "cuda"):
    """Load CLIP or SigLIP model and processor."""
    if "clip" in model_name.lower():
        print(f"Loading CLIP model: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(device)
        processor = CLIPProcessor.from_pretrained(model_name)
    else:
        print(f"Loading SigLIP model: {model_name}")
        model = AutoModel.from_pretrained(model_name).to(device)
        processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def encode_image(model, processor, image: np.ndarray, device: str, is_clip: bool):
    """Encode a single image to embedding. Image is (H, W, 3) uint8."""
    pil = Image.fromarray(image)
    if is_clip:
        inputs = processor(images=[pil], return_tensors="pt", padding=True)
    else:
        inputs = processor(images=[pil], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu()


def encode_text(model, processor, text: str, device: str, is_clip: bool):
    """Encode text to embedding."""
    if is_clip:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu()


def get_valid_positions_continuous(
    layout: np.ndarray,
    grid_resolution: int,
    x_min: float = 1.5,
    x_max: float = 14.5,
    y_min: float = 1.5,
    y_max: float = 14.5,
):
    """Return list of (x, y) float positions for continuous env where agent doesn't overlap walls."""
    x_vals = np.linspace(x_min, x_max, grid_resolution)
    y_vals = np.linspace(y_min, y_max, grid_resolution)
    valid = []
    for x in x_vals:
        for y in y_vals:
            # Agent occupies [x-0.5, x+0.5] x [y-0.5, y+0.5]; check overlapping cells
            i_min = max(0, int(np.floor(x - 0.5)))
            i_max = min(X_BOUND, int(np.ceil(x + 0.5)))
            j_min = max(0, int(np.floor(y - 0.5)))
            j_max = min(Y_BOUND, int(np.ceil(y + 0.5)))
            valid_pos = True
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    if layout[i, j] == "#":
                        valid_pos = False
                        break
                if not valid_pos:
                    break
            if valid_pos:
                valid.append((float(x), float(y)))
    return valid


def get_floor_render_data(layout: np.ndarray, pad_colors: dict = None) -> dict:
    """Build floor data for visualization - walls and colored pads (or white for open)."""
    if pad_colors is None:
        pad_colors = {}
    elements = []

    # Background
    elements.append({
        "type": "mesh3d",
        "x": [0.0, float(X_BOUND), float(X_BOUND), 0.0],
        "y": [0.0, 0.0, float(Y_BOUND), float(Y_BOUND)],
        "z": [-0.1, -0.1, -0.1, -0.1],
        "color": "white",
        "opacity": 1.0,
        "name": "Background",
    })

    # Grid cells as small quads
    for x in range(X_BOUND):
        for y in range(Y_BOUND):
            char = layout[x, y]
            z_val = -0.05
            if char == "#":
                color = "rgb(192, 192, 192)"
            elif char in pad_colors:
                rgb = pad_colors[char]
                color = f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
            else:
                color = "rgb(255, 255, 255)"

            x_vals = [float(x), float(x + 1), float(x + 1), float(x)]
            y_vals = [float(y), float(y), float(y + 1), float(y + 1)]
            z_vals = [z_val] * 4

            elements.append({
                "type": "mesh3d",
                "x": x_vals,
                "y": y_vals,
                "z": z_vals,
                "i": [0, 0],
                "j": [1, 2],
                "k": [2, 3],
                "color": color,
                "opacity": 1.0,
                "name": "cell",
            })

    return {"elements": elements}


def run_heatmap(
    goal_prompts: list = None,
    goal_positions: list = None,
    model_name: str = "openai/clip-vit-base-patch32",
    output_dir: str = "results/pinpad_heatmap",
    device: str = "cuda",
    batch_size: int = 32,
    grid_resolution: int = 64,
):
    if goal_prompts is None:
        goal_prompts = [
            "laser pointer on the cat's eye",
            "laser pointer on the cat's ear",
            "laser pointer on the cat's paws",
            "laser pointer on the cat's nose",
        ]
    if goal_positions is None:
        # Default target points for PinPadImage (x, y in [1.5, 14.5])
        # Approximate: eye, ear, paws, nose
        goal_positions = [[8.0, 5.0], [5.0, 3.0], [7.0, 12.0], [8.0, 7.0]]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Create PinPadImage env (image background, red dot agent)
    import stable_worldmodel.envs  # noqa: F401 - registers envs
    from stable_worldmodel.envs.pinpad.constants import LAYOUT_OPEN

    env = gym.make("swm/PinPadImage-v0")
    obs, info = env.reset(seed=42)
    layout = env.unwrapped.layout

    # Continuous env: sample at finer grid in [1.5, 14.5]
    valid_positions = get_valid_positions_continuous(layout, grid_resolution)
    print(f"PinPadImage, Grid resolution: {grid_resolution}, Valid positions: {len(valid_positions)}")

    # Build 2D grid for heatmap: Z[i,j] = value at (x_vals[j], y_vals[i])
    x_vals = np.linspace(1.5, 14.5, grid_resolution)
    y_vals = np.linspace(1.5, 14.5, grid_resolution)
    pos_to_grid = {}  # (x,y) -> (i,j) where Z[i,j] corresponds to (x_vals[j], y_vals[i])
    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            pos_to_grid[(float(x), float(y))] = (i, j)

    # 2. Load model
    is_clip = "clip" in model_name.lower()
    model, processor = load_embedding_model(model_name, device)

    # 3. Embed all positions (same env, only player moves)
    print("Embedding grid positions (continuous env)...")
    all_images = []
    for (x, y) in tqdm(valid_positions, desc="Rendering"):
        img = env.unwrapped.render(player_position=(x, y))
        all_images.append(img)

    # Process in batches
    all_embeddings = []
    for i in tqdm(range(0, len(all_images), batch_size), desc="Encoding"):
        batch = all_images[i : i + batch_size]
        pil_list = [Image.fromarray(im) for im in batch]
        if is_clip:
            inputs = processor(images=pil_list, return_tensors="pt", padding=True)
        else:
            inputs = processor(images=pil_list, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu())
    grid_embeddings = torch.cat(all_embeddings, dim=0)

    # Reference env image for left panel (agent at center)
    env_ref_img = env.unwrapped.render(player_position=(8.0, 8.0))
    env_agent_center_path = output_path / "pinpad_image_env_agent_center.png"
    Image.fromarray(env_ref_img).save(env_agent_center_path)
    print(f"Saved environment (agent at center) to {env_agent_center_path}")

    # Environment with no agent (player off-screen) - for use as floor/bottom of graph
    env_no_agent_img = env.unwrapped.render(player_position=(-10.0, -10.0))
    env_no_agent_path = output_path / "pinpad_image_env_no_agent.png"
    Image.fromarray(env_no_agent_img).save(env_no_agent_path)
    print(f"Saved environment (no agent) to {env_no_agent_path}")

    # 4. For each goal prompt, compute distances and save
    for goal_idx, prompt in enumerate(goal_prompts):
        print(f"Processing goal {goal_idx}: '{prompt}'")
        text_emb = encode_text(model, processor, prompt, device, is_clip)
        text_emb = text_emb.unsqueeze(0)
        dists = torch.norm(grid_embeddings - text_emb, dim=1).numpy()

        # Fill 2D distance grid: Z[i,j] at (x_vals[j], y_vals[i])
        Z = np.full((grid_resolution, grid_resolution), np.nan)
        for idx, (x, y) in enumerate(valid_positions):
            if (x, y) in pos_to_grid:
                i, j = pos_to_grid[(x, y)]
                Z[i, j] = dists[idx]

        # --- Save side-by-side: env on left, heatmap on right ---
        fig, (ax_env, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Sample PinPadImage environment
        ax_env.imshow(env_ref_img, extent=[0, X_BOUND, Y_BOUND, 0], origin="upper")
        ax_env.set_title("PinPadImage (cat_background.jpg, red dot agent)")
        ax_env.set_xlim(0, X_BOUND)
        ax_env.set_ylim(Y_BOUND, 0)
        ax_env.set_aspect("equal")
        ax_env.axis("on")

        # Right: CLIP L2 distance heatmap
        # Z[i,j] = value at (x_vals[j], y_vals[i]); with origin='upper', row 0 = top = y_min
        Z_display = np.ma.masked_where(np.isnan(Z), Z)
        im = ax_heat.imshow(
            Z_display,
            extent=[1.5, 14.5, 14.5, 1.5],
            origin="upper",
            cmap="viridis",
            aspect="equal",
            vmin=np.nanmin(Z),
            vmax=np.nanmax(Z),
        )
        model_label = "SigLIP 2" if "siglip" in model_name.lower() else "CLIP"
        ax_heat.set_xlabel("X-Coordinate of Center of Agent")
        ax_heat.set_ylabel("Y-Coordinate of Center of Agent")
        ax_heat.set_title(f"{model_label} L2 Distance Heatmap")
        ax_heat.set_xlim(1.5, 14.5)
        ax_heat.set_ylim(14.5, 1.5)
        ax_heat.grid(True, alpha=0.3)
        cbar = plt.colorbar(im, ax=ax_heat, label="L2 Distance")
        ax_heat.text(
            0.02, 0.98, f'"{prompt[:45]}{"..." if len(prompt) > 45 else ""}"',
            transform=ax_heat.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        png_path = output_path / f"heatmap_goal_{goal_idx}.png"
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved {png_path}")

        # --- Save JSON for 3D visualization (like latent_viz) ---
        scale_factor = 0.5
        floor_z = float(np.nanmin(Z) * scale_factor - 0.5) if not np.all(np.isnan(Z)) else -1.0

        # Plotly surface: z[i][j] at (x[j], y[i])
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z_grid = np.where(np.isnan(Z), 0, Z * scale_factor)

        floor_data = get_floor_render_data(layout)

        viz_data = {
            "metadata": {
                "environment_name": "pinpad_image",
                "goal_prompt": prompt,
                "goal_index": goal_idx,
                "goal_position": goal_positions[goal_idx] if goal_idx < len(goal_positions) else None,
                "model": model_name,
                "distance_stats": {
                    "min": float(np.nanmin(Z)),
                    "max": float(np.nanmax(Z)),
                    "mean": float(np.nanmean(Z)),
                },
            },
            "surface_data": {
                "x": X_grid.tolist(),
                "y": Y_grid.tolist(),
                "z": Z_grid.tolist(),
                "colorscale": "Viridis",
                "opacity": 0.8,
            },
            "floor_data": floor_data,
            "layout": {
                "title": f"PinPadImage - Text-Image Distance: \"{prompt[:40]}...\"",
                "scene": {
                    "xaxis_title": "X",
                    "yaxis_title": "Y",
                    "zaxis_title": "L2 Distance",
                    "aspectmode": "data",
                },
                "width": 1200,
                "height": 900,
            },
        }

        json_path = output_path / f"heatmap_goal_{goal_idx}.json"
        with open(json_path, "w") as f:
            json.dump(viz_data, f, indent=2)
        print(f"  Saved {json_path}")

    # Save embeddings for reuse (include model in filename to avoid overwriting)
    model_short = "siglip2" if "siglip" in model_name.lower() else "clip"
    save_data = {
        "embeddings": grid_embeddings,
        "valid_positions": valid_positions,
        "env_type": "pinpad_image",
        "goal_positions": goal_positions,
        "model": model_name,
        "layout": layout.tolist(),
    }
    torch.save(save_data, output_path / f"grid_embeddings_{model_short}.pt")
    print(f"Saved grid embeddings to {output_path / f'grid_embeddings_{model_short}.pt'}")

    env.close()
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="PinPadImage CLIP/SigLIP heatmap visualization")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--goals",
        type=str,
        nargs="+",
        default=[
            "laser pointer on the cat's eye",
            "laser pointer on the cat's ear",
            "laser pointer on the cat's paws",
            "laser pointer on the cat's nose",
        ],
        help="Goal text prompts",
    )
    parser.add_argument(
        "--goal-positions",
        type=float,
        nargs="+",
        default=None,
        help="Goal (x,y) positions as flat list, e.g. 3 8 13 8 8 8 for 3 goals",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP or SigLIP model name",
    )
    parser.add_argument("--output", type=str, default="results/pinpad_heatmap")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grid-resolution", type=int, default=64, help="Samples per axis")
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        goals = cfg.get("goal_prompts", args.goals)
        goal_positions = cfg.get("goal_positions", args.goal_positions)
        model_name = cfg.get("model", args.model)
        output_dir = cfg.get("output_dir", args.output)
        grid_resolution = cfg.get("grid_resolution", args.grid_resolution)
    else:
        goals = args.goals
        goal_positions = args.goal_positions
        model_name = args.model
        output_dir = args.output
        grid_resolution = args.grid_resolution

    # Parse goal_positions: flat list [x1,y1, x2,y2, ...] -> [[x1,y1], [x2,y2], ...]
    if goal_positions is not None and len(goal_positions) >= 2:
        goal_positions = [
            [goal_positions[i], goal_positions[i + 1]]
            for i in range(0, len(goal_positions), 2)
        ]
    else:
        goal_positions = None

    run_heatmap(
        goal_prompts=goals,
        goal_positions=goal_positions,
        model_name=model_name,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        grid_resolution=grid_resolution,
    )


if __name__ == "__main__":
    main()
