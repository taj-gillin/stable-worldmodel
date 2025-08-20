## -- Policy
### BasePolicy
### RandomPolicy
### OptimalPolicy (Expert)
### PlanningPolicy (wm, solver)

import numpy as np
from einops import rearrange
import torch


class BasePolicy:
    """Base class for agent policies"""

    # a policy takes in an environment and a planner
    def __init__(self, env, horizon=1, **kwargs):
        self.env = env
        self.horizon = horizon

    def get_action(self, obs, goal_obs, **kwargs):
        """Get action from the policy given the observation"""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def get_action(self, obs, goal_obs, **kwargs):
        action_seq = []
        for step in range(self.horizon):
            action = self.env.action_space.sample()
            action_seq.append(action)

        action_seq = np.stack(action_seq, axis=1)

        if action_seq.ndim == 2:
            action_seq = action_seq[:, np.newaxis, :]

        return action_seq


class OptimalPolicy(BasePolicy):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def get_action(self, obs, goal_obs, **kwargs):
        # Implement optimal policy logic here
        pass


class PlanningPolicy(BasePolicy):
    def __init__(self, env, planning_solver, output_dir="./results", **kwargs):
        super().__init__(env, **kwargs)
        self.solver = planning_solver
        self.output_dir = output_dir

    def get_action(self, obs, goal_obs, decode=False, **kwargs):
        actions = self.solver(obs, self.env.action_space, goal_obs)

        if decode:
            self.dump_decoded_trajectories(obs, actions, video=True)

        return actions

    def dump_decoded_trajectories(self, obs_0, actions, video=False):
        import imageio

        wm = self.solver.unwrapped.world_model

        if wm is None:
            raise ValueError("World model is None, cannot debug imagined trajectories")

        # expect post solver actions
        actions = torch.tensor(actions)
        actions = wm.normalize_actions(actions).to(obs_0["proprio"].device)
        actions = rearrange(actions, "b (t f) d -> b t (f d)", f=wm.frameskip)

        # -- simulate the world under actions sequence
        z_obs_i, z = wm.rollout(obs_0, actions)

        # -- decode obs
        decoded_obs, _ = wm.decode_obs(z_obs_i)

        for env_idx, traj in enumerate(decoded_obs["pixels"]):
            env_output_dir = self.output_dir / f"env_{env_idx}"
            env_output_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            for idx, frame in enumerate(traj):
                frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                frame = frame.detach().cpu().numpy()
                frames.append(frame)

            if video:
                video_path = env_output_dir / f"imagined_{env_idx}.mp4"
                video_writer = imageio.get_writer(video_path, fps=12)

            for idx, frame in enumerate(frames):
                frame = frame * 2 - 1 if frame.min() >= 0 else frame
                frame = (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                if video:
                    video_writer.append_data(frame)

                # save the image
                img_path = env_output_dir / f"imagined_frame_{idx}.png"
                imageio.imwrite(img_path, frame)

            if video:
                video_writer.close()
