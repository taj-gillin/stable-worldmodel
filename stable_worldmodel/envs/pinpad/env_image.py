"""PinPadImage: Image background with red laser-pointer agent.

Same movement as PinPad (continuous 2D), but the background is an image
and the agent is rendered as a red dot (laser pointer / flashlight).
Goal logic is not implemented yet; eventually the goal will be to navigate
to different parts of the image.
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from PIL import Image

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.pinpad.constants import (
    X_BOUND,
    Y_BOUND,
    RENDER_SCALE,
    LAYOUT_OPEN,
)


DEFAULT_VARIATIONS = ('agent.spawn',)


def _load_background_image(path: Path, width: int, height: int) -> np.ndarray:
    """Load and resize background image to (width, height). Returns RGB array."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img.resize((width, height), Image.Resampling.LANCZOS))


class PinPadImage(gym.Env):
    """PinPad variant: image background + red laser-pointer agent.

    Same action/observation spaces and movement as PinPad. The agent is
    a small red dot (laser pointer style). Goal is not implemented yet.
    """

    def __init__(
        self,
        background_image_path=None,
        seed=None,
        init_value=None,
        render_mode='rgb_array',
    ):
        self.background_image_path = background_image_path
        if background_image_path is None:
            assets_dir = Path(__file__).parent / "assets"
            self.background_image_path = assets_dir / "cat_background.jpg"

        self.variation_space = self._build_variation_space()
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        self.observation_space = gym.spaces.Dict(
            {
                'image': gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(Y_BOUND * RENDER_SCALE, X_BOUND * RENDER_SCALE, 3),
                    dtype=np.uint8,
                ),
                'agent_position': gym.spaces.Box(
                    low=np.array([1.5, 1.5], dtype=np.float64),
                    high=np.array(
                        [X_BOUND - 1.5, Y_BOUND - 1.5], dtype=np.float64
                    ),
                    shape=(2,),
                    dtype=np.float64,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64,
        )

        self.layout = None
        self.player = None
        self._background_cache = None

    def _build_variation_space(self):
        return swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'spawn': swm_spaces.Box(
                            low=np.array([1.5, 1.5], dtype=np.float64),
                            high=np.array(
                                [X_BOUND - 1.5, Y_BOUND - 1.5],
                                dtype=np.float64,
                            ),
                            init_value=np.array(
                                [X_BOUND / 2, Y_BOUND / 2], dtype=np.float64
                            ),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                    }
                ),
            },
            sampling_order=['agent'],
        )

    def _setup_layout(self):
        self.layout = np.array(
            [list(line) for line in LAYOUT_OPEN.split('\n')]
        ).T
        assert self.layout.shape == (X_BOUND, Y_BOUND)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        options = options or {}
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed,
            options,
            DEFAULT_VARIATIONS,
        )

        if self.layout is None:
            self._setup_layout()

        spawn_position = self.variation_space['agent']['spawn'].value
        self.player = tuple(spawn_position)

        # Invalidate background cache if path changed
        self._background_cache = None

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self):
        return {
            'image': self.render(),
            'agent_position': np.array(self.player, dtype=np.float64),
        }

    def step(self, action):
        x = np.clip(self.player[0] + action[0], 1.5, X_BOUND - 1.5)
        y = np.clip(self.player[1] + action[1], 1.5, Y_BOUND - 1.5)
        tile = self.layout[int(x)][int(y)]
        if tile != '#':
            self.player = (float(x), float(y))

        obs = self._get_obs()
        reward = 0.0  # No goal yet, I'm thinking we'll probably just give reward when in the target space
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        return {}

    def render(self, player_position=None):
        if player_position is None:
            player_position = self.player

        height = Y_BOUND * RENDER_SCALE
        width = X_BOUND * RENDER_SCALE

        # Load/cache background
        if self._background_cache is None:
            path = Path(self.background_image_path)
            self._background_cache = _load_background_image(
                path, width, height
            ).copy()

        image = self._background_cache.copy()

        # Draw walls
        for (x, y), char in np.ndenumerate(self.layout):
            if char == '#':
                px_min, px_max = x * RENDER_SCALE, (x + 1) * RENDER_SCALE
                py_min, py_max = y * RENDER_SCALE, (y + 1) * RENDER_SCALE
                image[py_min:py_max, px_min:px_max] = (192, 192, 192)

        # Draw red laser-pointer dot
        dot_radius = max(2, RENDER_SCALE // 3)  
        cx = int(player_position[0] * RENDER_SCALE)
        cy = int(player_position[1] * RENDER_SCALE)

        y_min = max(0, cy - dot_radius)
        y_max = min(height, cy + dot_radius + 1)
        x_min = max(0, cx - dot_radius)
        x_max = min(width, cx + dot_radius + 1)

        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                if (px - cx) ** 2 + (py - cy) ** 2 <= dot_radius ** 2:
                    image[py, px] = (255, 0, 0)

        return image
