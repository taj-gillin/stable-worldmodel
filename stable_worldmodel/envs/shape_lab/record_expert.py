#!/usr/bin/env python3
"""
Record expert policy trajectories as videos.

Usage:
    python -m stable_worldmodel.envs.shape_lab.record_expert
    python -m stable_worldmodel.envs.shape_lab.record_expert --seed 123 --max-steps 2000
    python -m stable_worldmodel.envs.shape_lab.record_expert --output my_video.mp4 --fps 60
"""

import argparse
import os

from stable_worldmodel.envs.shape_lab.env import ShapeLab
from stable_worldmodel.envs.shape_lab.expert import ShapeLabExpert


def record_expert_video(
    output_path: str = "expert_trajectory.mp4",
    seed: int = 42,
    max_steps: int = 3000,
    fps: int = 30,
    resolution: int = 512,
    frame_skip: int = 2,
    debug: bool = False,
    start_round: int = 0,
) -> dict:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    env = ShapeLab(render_mode="rgb_array", resolution=resolution)
    expert = ShapeLabExpert(env, debug=debug)

    expert.reset()
    reset_options = {"start_round": start_round} if start_round > 0 else {}
    obs, info = env.reset(seed=seed, options=reset_options)

    print(f"Recording expert trajectory (seed={seed}, max_steps={max_steps}, start_round={start_round})")

    frames = []
    total_reward = 0.0
    rounds_completed = 0
    step = 0

    for step in range(max_steps):
        action = expert.get_action(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % frame_skip == 0:
            frames.append(obs["image"])

        if reward > 0.5:
            rounds_completed += 1
            if debug:
                print(f"  Round completed! Total: {rounds_completed}")

        if terminated or truncated:
            break

    env.close()

    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"\nSaved {len(frames)} frames to {output_path}")
    except ImportError:
        print("\nWarning: imageio not installed. Saving frames as PNG instead.")
        frames_dir = output_path.replace(".mp4", "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        from PIL import Image
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(f"{frames_dir}/frame_{i:05d}.png")
        print(f"Saved {len(frames)} frames to {frames_dir}/")

    stats = {
        "steps": step + 1,
        "total_reward": total_reward,
        "rounds_completed": rounds_completed,
        "frames": len(frames),
        "output_path": output_path,
    }

    print(f"\nSteps: {stats['steps']} | Reward: {stats['total_reward']:.2f} | Rounds: {stats['rounds_completed']}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Record expert policy trajectories as videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output", type=str, default="test_videos/expert_trajectory.mp4")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-n", "--max-steps", type=int, default=3000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-r", "--start-round", type=int, default=0)

    args = parser.parse_args()
    record_expert_video(
        output_path=args.output,
        seed=args.seed,
        max_steps=args.max_steps,
        fps=args.fps,
        resolution=args.resolution,
        frame_skip=args.frame_skip,
        debug=args.debug,
        start_round=args.start_round,
    )


if __name__ == "__main__":
    main()
