from pathlib import Path
import torch
from .policy import BasePolicy
from .world import World
import numpy as np


## -- Evaluator / Collector
### Evaluator(env, policy)
class Evaluator:
    # the role of evaluator is to determine perf of the policy in the env
    def __init__(
        self, world: World, policy: BasePolicy, output_dir="./results", device="cpu"
    ):
        self.world = world
        self.policy = policy
        self.device = device
        self.output_dir = output_dir

    # TODO move ths to the policy class
    def prepare_obs(self, obs):
        """Prepare observations for the policy."""
        # torchify observations and move to device
        obs = {k: torch.from_numpy(v).to(self.device) for k, v in obs.items()}
        # unbind the temporal dimension
        obs = {k: v.unsqueeze(1) for k, v in obs.items()}
        return obs

    def run(self, episodes=1):
        # todo return interested logging data
        data = {}

        for episode in range(episodes):
            actions = np.empty((0), dtype=np.float32)
            for obs, goal_obs, rewards in self.world:
                # preprocess obs for pytorch
                obs = self.prepare_obs(obs)
                goal_obs = self.prepare_obs(goal_obs)

                # -- get actions from the policy
                if actions.size == 0:
                    actions = self.policy.get_action(obs, goal_obs, decode=True)

                exec_action, actions = actions[:, 0], actions[:, 1:]

                # actions = actions.squeeze(0) if actions.ndim == 2 else actions
                # apply actions in the env
                # for a in actions.unbind(0):
                #     self.world.step(a.numpy())

                # make actions double precision (np array)
                exec_action = (
                    exec_action.double().numpy()
                    if isinstance(exec_action, torch.Tensor)
                    else exec_action
                )

                # print(obs["proprio"].cpu().numpy())
                # print(goal_obs["proprio"].cpu().numpy())
                # print("===============")

                # print(exec_action)

                # ! keep
                # assert action is between -1 and 1
                # assert np.all(np.abs(exec_action) <= 1.0), "Action out of bound [-1, 1]"

                # TODO SHOULD GET SOME DATA FROM THE ENV TO KNOW HOW GOOD
                self.world.step(exec_action)

            print(f"Episode {episode + 1} finished ")

            goal_obs = goal_obs["state"].squeeze(1).cpu().numpy()
            obs = obs["state"].squeeze(1).cpu().numpy()
            self.eval_state(goal_obs, obs)

            self.world.close()

        return data

    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        from: https://github.com/gaoyuezhou/dino_wm/blob/main/env/pusht/pusht_wrapper.py
        """

        # if position difference is < 20, and angle difference < np.pi/9, then success
        pos_diff = np.linalg.norm(
            goal_state[:, :4] - cur_state[:, :4], axis=-1
        )  # (batch_size,)
        angle_diff = np.abs(goal_state[:, 4] - cur_state[:, 4])  # (batch_size,)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # (batch_size,)

        success = (pos_diff < 20) & (angle_diff < np.pi / 9)  # (batch_size,)
        state_dist = np.linalg.norm(goal_state - cur_state, axis=-1)  # (batch_size,)

        for i in range(len(success)):
            env_output_dir = self.output_dir / f"env_{i}"
            env_output_dir.mkdir(parents=True, exist_ok=True)
            with open(env_output_dir / "results.txt", "a") as f:
                f.write(f"Succes: {success[i]}\nState distance: {state_dist[i]}\n")

        return success, state_dist
