from pathlib import Path
import torch
from .policy import BasePolicy
from .world import World
import numpy as np
import json


## -- Evaluator / Collector
### Evaluator(env, policy)
class Evaluator:
    # the role of evaluator is to determine perf of the policy in the env
    def __init__(
        self, world: World, policy: BasePolicy, device="cpu", output_dir="results"
    ):
        self.world = world
        self.policy = policy
        self.device = device
        self.successes = np.zeros((world.unwrapped.num_envs,), dtype=np.bool)
        self._actions = np.empty((world.unwrapped.num_envs, 0), dtype=np.float32)
        self.output_dir = Path(output_dir)

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
            for obs, goal_obs, rewards in self.world:

                # eval current state
                successes, _ = self.eval_state(goal_obs["state"], obs["state"])

                # update successes
                self.successes = self.successes | successes

                # preprocess obs for pytorch
                obs = self.prepare_obs(obs)
                goal_obs = self.prepare_obs(goal_obs)
     
                # -- get actions from the policy
                if self._actions.shape[1] == 0:
                    self._actions = self.policy.get_action(obs, goal_obs, decode=True)
                    print(self._actions.shape)

                exec_action, self._actions = self._actions[:, 0], self._actions[:, 1:]

                print(exec_action.shape)
                print(self._actions.shape)

                # actions = actions.squeeze(0) if actions.ndim == 2 else actions
                # apply actions in the env
                # for a in actions.unbind(0):
                #     self.world.step(a.numpy()

                # make actions double precision (np array)
                exec_action = (
                    exec_action.double().numpy()
                    if isinstance(exec_action, torch.Tensor)
                    else exec_action
                )

                # masked actions from success env
                exec_action[self.successes] = 0.0

                print(self.successes)

                # print(obs["proprio"].cpu().numpy())
                # print(goal_obs["proprio"].cpu().numpy())
                # print("===============")

                # print(exec_action)

                # ! keep
                # assert action is between -1 and 1
                # assert np.all(np.abs(exec_action) <= 1.0), "Action out of bound [-1, 1]"

                # TODO SHOULD GET SOME DATA FROM THE ENV TO KNOW HOW GOOD
                self.world.step(exec_action)

                if self.successes.all():
                    print(f"Episode {episode + 1} finished successfully")
                    break

            print(f"Episode {episode + 1} finished ")

            self.world.close()

        return data

    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        from: https://github.com/gaoyuezhou/dino_wm/blob/main/env/pusht/pusht_wrapper.py
        """

        # if position difference is < 20, and angle difference < np.pi/9, then success
        pos_diff = np.linalg.norm(goal_state[:, :4] - cur_state[:, :4], axis=-1)  # (batch_size,)
        angle_diff = np.abs(goal_state[:, 4] - cur_state[:, 4])  # (batch_size,)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # (batch_size,)

        success = (pos_diff < 20) & (angle_diff < np.pi / 9)  # (batch_size,)
        state_dist = np.linalg.norm(goal_state - cur_state, axis=-1)  # (batch_size,)

        results = {
            str(i): {
                "success": bool(s),
                "state_distance": float(d),
            }
            for i, (s, d) in enumerate(zip(success, state_dist))
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        return success, state_dist
