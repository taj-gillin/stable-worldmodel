"""
PinPad: OPRO-style prompt optimization for correlation with ground-truth distance.

Maximizes Pearson correlation between:
- L2 embedding distance (image vs goal text prompt)
- Ground-truth distance (agent position to closest point on goal square)

Usage:
    python scripts/visualization/pinpad_optimize_prompt.py \
        --embeddings results/pinpad_heatmap_siglip2/grid_embeddings_siglip2.pt \
        --goal-idx 0 \
        --initial-prompt "black dot on a blue square"

Requires grid_embeddings from pinpad_heatmap.py. Run heatmap first if needed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import openai
from scipy import stats

# Allow imports from same directory
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import gymnasium as gym
import stable_worldmodel.envs  # noqa: F401
from stable_worldmodel.envs.pinpad.constants import X_BOUND, Y_BOUND

from pinpad_distance_utils import (
    compute_ground_truth_distances,
    compute_ground_truth_distances_to_point,
    get_goal_bounds_for_task,
)
from pinpad_correlation_analysis import goal_prompt_to_pad, plot_correlation
from pinpad_heatmap import load_embedding_model, encode_text


class PinPadCorrelationScorer:
    """Scores a prompt by Pearson correlation between L2 distance and ground-truth distance."""

    def __init__(
        self,
        embeddings_path: str,
        goal_idx: int,
        goal_prompt: str,
        task: str = "three",
        model_name: str = "google/siglip2-so400m-patch14-384",
        device: str = "cuda",
    ):
        self.device = device
        self.goal_idx = goal_idx
        self.goal_prompt = goal_prompt
        self.task = task
        self.model_name = model_name
        self.embeddings_path = embeddings_path

        data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
        self.positions = [(float(x), float(y)) for x, y in data["valid_positions"]]
        self.embeddings = data["embeddings"]
        env_type = data.get("env_type", "pinpad")
        goal_positions = data.get("goal_positions", [])

        if env_type == "pinpad_image" and goal_positions and goal_idx < len(goal_positions):
            goal_point = tuple(goal_positions[goal_idx])
            self.gt_dist = compute_ground_truth_distances_to_point(self.positions, goal_point)
            self.pad = str(goal_point)
        else:
            bounds = get_goal_bounds_for_task(task)
            self.pad = goal_prompt_to_pad(goal_prompt, bounds) or (list(bounds.keys())[0] if bounds else "1")
            self.gt_dist = compute_ground_truth_distances(self.positions, bounds[self.pad])

        print(f"Loading model: {model_name}")
        self.model, self.processor = load_embedding_model(model_name, device)
        self.is_clip = "clip" in model_name.lower()

    def get_distances(self, query: str) -> np.ndarray:
        text_emb = encode_text(self.model, self.processor, query, self.device, self.is_clip)
        text_emb = text_emb.unsqueeze(0).to(self.device)
        dists = torch.norm(self.embeddings.to(self.device) - text_emb, dim=1)
        return dists.cpu().numpy()

    def score(self, query: str, metric: str = "pearson") -> float:
        l2_dist = self.get_distances(query)
        if metric == "pearson":
            corr, _ = stats.pearsonr(self.gt_dist, l2_dist)
        elif metric == "spearman":
            corr, _ = stats.spearmanr(self.gt_dist, l2_dist)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return corr


class OPROOptimizer:
    """OPRO-style prompt optimizer for PinPad."""

    def __init__(
        self,
        scorer: PinPadCorrelationScorer,
        api_key: str = None,
        model: str = "gpt-4o-mini",
    ):
        self.scorer = scorer
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.history: list[tuple[str, float]] = []

    def generate_prompts(self, num_prompts: int = 5) -> list[str]:
        sorted_history = sorted(self.history, key=lambda x: x[1], reverse=True)
        top_history = sorted_history[:20]
        history_str = "\n".join([f'Prompt: "{p}", Correlation: {s:.4f}' for p, s in top_history])

        system_prompt = """You are an optimizer for vision-language model text prompts.

The environment is PinPadImage: an image background (e.g. a cat scene) with a red dot (laser pointer) as the agent that moves around.
For each goal, we have a target position and a text prompt describing it (e.g. "red dot on the left" or "red dot in the center").
We render an image at each grid position and compute the L2 distance between the image embedding and the goal prompt embedding.

Ground truth: Euclidean distance from each position to the target point. At the target, distance=0. Farther away, distance increases.

We want the L2 embedding distance to CORRELATE with ground-truth distance: as you get farther from the goal position, L2 should increase.

Your task is to propose {num_prompts} NEW text prompts that may improve this correlation. Be descriptive about which part of the cat the laser pointer is on (eye, ear, paws, nose, tail, etc.).
Output ONLY a JSON list of strings, e.g. ["prompt 1", "prompt 2"]."""

        user_prompt = f"""History of prompts and Pearson correlation scores:
{history_str}

Goal: MAXIMIZE the correlation. Propose {num_prompts} new prompts.
Output a valid JSON list of strings."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt.format(num_prompts=num_prompts)},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            if isinstance(data, list):
                prompts = data
            elif isinstance(data, dict):
                prompts = next((v for v in data.values() if isinstance(v, list)), [])
            else:
                prompts = [str(data)]
            return [str(p) for p in prompts[:num_prompts]]
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []

    def step(self, prompt: str) -> float:
        score = self.scorer.score(prompt, metric="pearson")
        self.history.append((prompt, score))
        return score

    def optimize(self, initial_prompt: str, n_steps: int = 10, batch_size: int = 5):
        best_prompt = initial_prompt
        best_score = self.step(initial_prompt)
        print(f"Step 0: \"{initial_prompt[:50]}...\" -> Correlation={best_score:.4f}")

        for i in range(1, n_steps + 1):
            new_prompts = self.generate_prompts(num_prompts=batch_size)
            for prompt in new_prompts:
                score = self.step(prompt)
                print(f"  \"{prompt[:45]}...\" -> {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_prompt = prompt
                    print("  --> New best!")
        return best_prompt, best_score


def plot_history(history: list, output_path: Path):
    _, scores = zip(*history)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, marker="o", markersize=4)
    plt.xlabel("Step")
    plt.ylabel("Pearson Correlation")
    plt.title("PinPad Prompt Optimization")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_correlation_and_heatmap(
    scorer: PinPadCorrelationScorer,
    best_prompt: str,
    output_dir: Path,
    goal_idx: int,
    task: str,
):
    """Save correlation scatter plot and L2 distance heatmap for the best prompt."""
    positions = scorer.positions
    l2_dist = scorer.get_distances(best_prompt)
    gt_dist = scorer.gt_dist
    pearson, _ = stats.pearsonr(gt_dist, l2_dist)
    spearman, _ = stats.spearmanr(gt_dist, l2_dist)
    goal_name = f"Goal {goal_idx} ({scorer.pad})"

    # Correlation graph
    corr_path = output_dir / f"correlation_goal_{goal_idx}.png"
    plot_correlation(gt_dist, l2_dist, corr_path, best_prompt, goal_name, pearson, spearman)
    print(f"  Saved correlation: {corr_path}")

    # JSON of points for web graphing
    points_data = {
        "metadata": {
            "goal_idx": goal_idx,
            "pad": scorer.pad,
            "prompt": best_prompt,
            "pearson": float(pearson),
            "spearman": float(spearman),
            "task": task,
        },
        "points": [
            {"x": float(x), "y": float(y), "gt_dist": float(gt), "l2_dist": float(l2)}
            for (x, y), gt, l2 in zip(positions, gt_dist, l2_dist)
        ],
    }
    points_path = output_dir / f"correlation_goal_{goal_idx}.json"
    with open(points_path, "w") as f:
        json.dump(points_data, f, indent=2)
    print(f"  Saved points JSON: {points_path}")

    # Heatmap: build Z grid from positions and dists (positions and l2_dist are aligned)
    x_vals = np.unique(np.round([p[0] for p in positions], 10))
    y_vals = np.unique(np.round([p[1] for p in positions], 10))
    x_to_j = {round(float(x), 10): j for j, x in enumerate(x_vals)}
    y_to_i = {round(float(y), 10): i for i, y in enumerate(y_vals)}
    grid_res = len(x_vals)
    Z = np.full((grid_res, grid_res), np.nan)
    for idx, (x, y) in enumerate(positions):
        xr, yr = round(float(x), 10), round(float(y), 10)
        j = x_to_j.get(xr)
        i = y_to_i.get(yr)
        if j is not None and i is not None:
            Z[i, j] = l2_dist[idx]

    # Create env for reference image
    env = gym.make("swm/PinPadImage-v0")
    env.reset(seed=42)
    env_ref_img = env.unwrapped.render(player_position=(8.0, 8.0))
    env.close()

    # Heatmap figure
    fig, (ax_env, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    ax_env.imshow(env_ref_img, extent=[0, X_BOUND, Y_BOUND, 0], origin="upper")
    ax_env.set_title("PinPadImage (cat_background.jpg, red dot agent)")
    ax_env.set_xlim(0, X_BOUND)
    ax_env.set_ylim(Y_BOUND, 0)
    ax_env.set_aspect("equal")
    ax_env.axis("on")

    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    Z_display = np.ma.masked_where(np.isnan(Z), Z)
    im = ax_heat.imshow(
        Z_display,
        extent=[x_min, x_max, y_max, y_min],
        origin="upper",
        cmap="viridis",
        aspect="equal",
        vmin=np.nanmin(Z),
        vmax=np.nanmax(Z),
    )
    model_label = "SigLIP 2" if "siglip" in scorer.model_name.lower() else "CLIP"
    ax_heat.set_xlabel("X-Coordinate of Center of Agent")
    ax_heat.set_ylabel("Y-Coordinate of Center of Agent")
    ax_heat.set_title(f"{model_label} L2 Distance Heatmap (Best Prompt)")
    ax_heat.set_xlim(x_min, x_max)
    ax_heat.set_ylim(y_max, y_min)
    ax_heat.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax_heat, label="L2 Distance")
    ax_heat.text(
        0.02, 0.98,
        f'"{best_prompt[:45]}{"..." if len(best_prompt) > 45 else ""}"',
        transform=ax_heat.transAxes, fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    heatmap_path = output_dir / f"heatmap_goal_{goal_idx}.png"
    plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {heatmap_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True, help="Path to grid_embeddings_*.pt")
    parser.add_argument("--goal-idx", type=int, default=0, help="Goal index (0=eye, 1=ear, 2=paws, 3=nose)")
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default="laser pointer on the cat's eye",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/pinpad_opro")
    parser.add_argument("--task", type=str, default="three")
    parser.add_argument("--model", type=str, default="google/siglip2-so400m-patch14-384")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    default_prompts = [
        "laser pointer on the cat's eye",
        "laser pointer on the cat's ear",
        "laser pointer on the cat's paws",
        "laser pointer on the cat's nose",
    ]
    goal_prompt = default_prompts[args.goal_idx % len(default_prompts)]

    if not os.path.exists(args.embeddings):
        print(f"ERROR: Embeddings not found: {args.embeddings}")
        print("Run pinpad_heatmap.py first to generate grid_embeddings.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    scorer = PinPadCorrelationScorer(
        args.embeddings,
        args.goal_idx,
        goal_prompt,
        task=args.task,
        model_name=args.model,
        device=args.device,
    )
    optimizer = OPROOptimizer(scorer)

    best_prompt, best_score = optimizer.optimize(
        args.initial_prompt or goal_prompt,
        n_steps=args.steps,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 50)
    print(f"Best prompt: \"{best_prompt}\"")
    print(f"Best correlation: {best_score:.4f}")
    print("=" * 50)

    results = {
        "goal_idx": args.goal_idx,
        "best_prompt": best_prompt,
        "best_score": float(best_score),
        "history": [(p, float(s)) for p, s in optimizer.history],
    }
    out_dir = Path(args.output_dir)
    with open(out_dir / f"opro_goal_{args.goal_idx}.json", "w") as f:
        json.dump(results, f, indent=2)
    plot_history(optimizer.history, out_dir / f"opro_goal_{args.goal_idx}.png")

    # Save correlation graph and heatmap for the best prompt
    print("\nSaving correlation and heatmap for best prompt...")
    save_correlation_and_heatmap(
        scorer, best_prompt, out_dir, args.goal_idx, args.task
    )
    print(f"Results saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
