import gymnasium as gym
import numpy as np
import torch
from loguru import logger as logging
from torchvision import transforms
from gymnasium.vector import SyncVectorEnv
from pathlib import Path

import xenoworlds


class World:
    def __init__(
        self,
        env_name,
        num_envs,
        wrappers: list = None,
        goal_wrappers: list = None,
        seed: int = 2349867,
        max_episode_steps: int = 100,
        sample_goal_every_k_steps: int = -1,
        output_dir: str = None,
    ):
        self.output_dir = output_dir
        self.envs, self.goal_envs = self.make_env(
            env_name,
            num_envs=num_envs,
            wrappers=wrappers,
            goal_wrappers=goal_wrappers,
            max_episode_steps=max_episode_steps,
            add_video_wrapper=output_dir is not None,
        )

        logging.info("WORLD INITIALIZED")
        logging.info(f"ACTION SPACE: {self.envs.action_space}")
        logging.info(f"OBSERVATION SPACE: {self.envs.observation_space}")
        self.num_envs = num_envs

        self.set_seed(seed)

        # note if sample_goal_every_k_steps is set to -1, will sample goal once per episode
        # TODO implement sample_goal_every_k_steps

    def make_env(
        self,
        env_name,
        num_envs=1,
        wrappers=(),
        goal_wrappers=(),
        max_episode_steps=100,
        add_video_wrapper=True,
    ):
        def build_env(extra_wrappers=(), idx=None):
            e = gym.make(
                env_name, render_mode="rgb_array", max_episode_steps=max_episode_steps
            )
            for w in extra_wrappers:
                e = w(e)
            if add_video_wrapper and idx is not None:
                env_output_dir = self.output_dir / f"env_{idx}"
                Path(env_output_dir).mkdir(parents=True, exist_ok=True)
                e = xenoworlds.wrappers.RecordVideo(
                    e, video_folder=env_output_dir, name_prefix=f"env_{idx}"
                )
            return e

        env_fns = [lambda i=i: build_env(wrappers, i) for i in range(num_envs)]
        goal_env_fns = [lambda: build_env(goal_wrappers, None) for _ in range(num_envs)]

        return SyncVectorEnv(env_fns), SyncVectorEnv(goal_env_fns)

    @property
    def observation_space(self):
        return self.envs.observation_space

    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    def close(self, **kwargs):
        return self.envs.close(**kwargs)

    # TEMOPORARY, need to delete!!!
    def denormalize(self, x):
        # x is (B,C,H,W) in [-1,1]
        return (x * 0.5) + 0.5

    def __iter__(self):
        self.terminations = np.array([False] * self.num_envs)
        self.truncations = np.array([False] * self.num_envs)
        self.rewards = None
        logging.info(f"Resetting the ({self.num_envs}) world(s)!")
        self.states, finfos = self.envs.reset(seed=self.env_seeds)
        self.goal_states, _ = self.goal_envs.reset(seed=self.goal_seeds)

        return self

    def __next__(self):
        if not all(self.terminations) and not all(self.truncations):
            return (
                self.states,
                self.goal_states,
                self.rewards,
            )
        else:
            raise StopIteration

    def step(self, actions):
        (
            self.states,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        ) = self.envs.step(actions)

    def set_seed(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.seed = seed

        # one seed per sub-env
        self.env_seeds = torch.randint(
            0, 2**32 - 1, (self.num_envs,), generator=rng
        ).tolist()
        self.goal_seeds = torch.randint(
            0, 2**32 - 1, (self.num_envs,), generator=rng
        ).tolist()


########## TMP #############


class PushTWorld(World):
    def __init__(
        self,
        env_name,
        num_envs,
        wrappers: list = None,
        goal_wrappers: list = None,
        seed: int = 2349867,
        max_episode_steps: int = 100,
        sample_goal_every_k_steps: int = -1,
        output_dir: str = None,
    ):
        self.output_dir = output_dir
        self.envs, self.goal_envs = self.make_env(
            env_name,
            num_envs=num_envs,
            wrappers=wrappers,
            goal_wrappers=goal_wrappers,
            max_episode_steps=max_episode_steps,
            add_video_wrapper=output_dir is not None,
        )

        logging.info("WORLD INITIALIZED")
        logging.info(f"ACTION SPACE: {self.envs.action_space}")
        logging.info(f"OBSERVATION SPACE: {self.envs.observation_space}")
        self.num_envs = num_envs

        self.set_seed(seed)

        # note if sample_goal_every_k_steps is set to -1, will sample goal once per episode
        # TODO implement sample_goal_every_k_steps

    def make_env(
        self,
        env_name,
        num_envs=1,
        wrappers=(),
        goal_wrappers=(),
        max_episode_steps=100,
        add_video_wrapper=True,
    ):
        def build_env(extra_wrappers=(), idx=None, is_goal=False):
            e = gym.make(
                env_name, render_mode="rgb_array", max_episode_steps=max_episode_steps
            )

            for w in extra_wrappers:
                try:
                    return w(e, idx, is_goal)
                except TypeError:
                    return w(e)

            if add_video_wrapper and not is_goal:
                env_output_dir = self.output_dir / f"env_{idx}"
                Path(env_output_dir).mkdir(parents=True, exist_ok=True)
                e = xenoworlds.wrappers.RecordVideo(
                    e, video_folder=env_output_dir, name_prefix=f"env_{idx}"
                )
            return e

        env_fns = [lambda i=i: build_env(wrappers, i, False) for i in range(num_envs)]
        goal_env_fns = [
            lambda i=i: build_env(goal_wrappers, i, True) for i in range(num_envs)
        ]

        return SyncVectorEnv(env_fns), SyncVectorEnv(goal_env_fns)

    @property
    def observation_space(self):
        return self.envs.observation_space

    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    def close(self, **kwargs):
        return self.envs.close(**kwargs)

    # TEMOPORARY, need to delete!!!
    def denormalize(self, x):
        # x is (B,C,H,W) in [-1,1]
        return (x * 0.5) + 0.5

    def __iter__(self):
        self.terminations = np.array([False] * self.num_envs)
        self.truncations = np.array([False] * self.num_envs)
        self.rewards = None
        logging.info(f"Resetting the ({self.num_envs}) world(s)!")
        self.states, finfos = self.envs.reset(seed=self.env_seeds)
        self.goal_states, _ = self.goal_envs.reset(seed=self.goal_seeds)

        return self

    def __next__(self):
        if not all(self.terminations) and not all(self.truncations):
            return (
                self.states,
                self.goal_states,
                self.rewards,
            )
        else:
            raise StopIteration

    def step(self, actions):
        (
            self.states,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        ) = self.envs.step(actions)

    def set_seed(self, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.seed = seed

        # one seed per sub-env
        self.env_seeds = torch.randint(
            0, 2**32 - 1, (self.num_envs,), generator=rng
        ).tolist()
        self.goal_seeds = torch.randint(
            0, 2**32 - 1, (self.num_envs,), generator=rng
        ).tolist()


############ Wrapppers #########


class WorldWrapper(World):
    def __init__(self, world):
        self.world = world

        if hasattr(world, "unwrapped"):
            self.unwrapped = world.unwrapped
        else:
            self.unwrapped = world

    @property
    def observation_space(self):
        return self.unwrapped.observation_space

    @property
    def action_space(self):
        return self.unwrapped.action_space

    @property
    def single_action_space(self):
        return self.unwrapped.single_action_space

    @property
    def single_observation_space(self):
        return self.unwrapped.single_observation_space

    def close(self, **kwargs):
        return self.unwrapped.close(**kwargs)

    def __iter__(self):
        iter(self.unwrapped)
        return self

    def __next__(self):
        return next(self.unwrapped)

    def step(self, actions):
        self.unwrapped.step(actions)

    def set_seed(self, seed):
        self.unwrapped.set_seed(seed)


class PushTRolloutCompletion(WorldWrapper):
    def __init__(self, world, horizon, dataset=None, transforms=None):
        super().__init__(world)
        self.world = world
        self.dataset = dataset
        self.horizon = horizon
        self.transforms = transforms or transforms.Compose([])

    def __iter__(self):
        states = {
            "proprio": [],
            "pixels": [],
            "state": [],
        }

        goals = {
            "proprio": [],
            "pixels": [],
            "state": [],
        }

        num_envs = self.unwrapped.num_envs

        for env_idx in range(num_envs):
            rng = np.random.default_rng(seed=self.unwrapped.env_seeds[env_idx])
            n_rollouts = len(self.dataset)
            rollout_idx = rng.integers(0, n_rollouts).item()
            rollout_length = self.dataset.get_seq_length(rollout_idx).item()
            rollout = self.dataset[rollout_idx]
            obs, state = rollout[0], rollout[2]

            # sample a random starting point in the rollout
            start_idx = rng.integers(0, rollout_length - self.horizon)

            #     init_obs = {
            #         "proprio": obs["proprio"][start_idx].numpy(),
            #         "pixels": obs["visual"][start_idx],
            #         "state": state[start_idx].numpy(),
            #     }

            #     goal_obs = {
            #         "proprio": obs["proprio"][start_idx + self.horizon].numpy(),
            #         "pixels": obs["visual"][start_idx + self.horizon],
            #         "state": state[start_idx + self.horizon].numpy(),
            #     }

            #     # apply transforms to visual observations
            #     init_obs["pixels"] = self.transforms(init_obs["pixels"]).numpy()
            #     goal_obs["pixels"] = self.transforms(goal_obs["pixels"]).numpy()

            #     states["proprio"].append(init_obs["proprio"])
            #     states["pixels"].append(init_obs["pixels"])
            #     states["state"].append(init_obs["state"])

            #     goals["proprio"].append(goal_obs["proprio"])
            #     goals["pixels"].append(goal_obs["pixels"])
            #     goals["state"].append(goal_obs["state"])

            #     init_state = np.array(state[start_idx], dtype=np.float64)
            #     self.unwrapped.envs.envs[env_idx].unwrapped._set_state(init_state)

            # self.unwrapped.states = {
            #     "proprio": np.stack(states["proprio"]),
            #     "pixels": np.stack(states["pixels"]),
            #     "state": np.stack(states["state"]),
            # }

            # self.unwrapped.goal_states = {
            #     "proprio": np.stack(goals["proprio"]),
            #     "pixels": np.stack(goals["pixels"]),
            #     "state": np.stack(goals["state"]),
            # }

            init_state = state[start_idx].numpy()
            goal_state = state[start_idx + self.horizon].numpy()

            self.unwrapped.envs.envs[env_idx].unwrapped.reset_to_state = init_state
            self.unwrapped.goal_envs.envs[env_idx].unwrapped.reset_to_state = goal_state

        print("Resetting the world with expert data!")

        iter(self.world)

        return self


class SaveInitAndGoal(WorldWrapper):
    def __init__(self, world):
        super().__init__(world)

    def __iter__(self):
        iter(self.world)

        def denormalize(x):
            # ! dino-wm specific
            # x is (C,H,W) in [-1,1]
            return (x * 0.5) + 0.5

        def _dump_obs(self, obs, prefix=""):
            """Save the initial and goal observations."""

            obs_pixels = obs["pixels"]  # (n_envs, C, H, W)
            obs_pixels = obs_pixels.transpose(0, 2, 3, 1)
            obs_pixels = (denormalize(obs_pixels) * 255.0).astype(np.uint8)
            to_pil = transforms.ToPILImage()

            for env_idx in range(self.unwrapped.num_envs):
                env_output_dir = self.unwrapped.output_dir / f"env_{env_idx}"
                env_output_dir.mkdir(parents=True, exist_ok=True)
                img = to_pil(obs_pixels[env_idx])
                # TODO add episode number if multiple episodes
                img.save(env_output_dir / f"{prefix}{env_idx}.png")

        _dump_obs(self, self.unwrapped.goal_states, prefix="goal_")
        _dump_obs(self, self.unwrapped.states, prefix="init_")

        return self
