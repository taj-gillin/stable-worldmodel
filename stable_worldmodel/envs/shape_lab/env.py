import os

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import stable_worldmodel as swm

from .entities import Color, DeliveryZone, PhysicsConfig, Player, ShapeType
from .game_logic import (
    CollisionHandler,
    RoundManager,
    add_walls,
    create_default_layout,
    draw_dispenser,
    draw_player,
    draw_shape,
    draw_tool,
    draw_ui,
    setup_physics_space,
)


class ShapeLab(gym.Env):
    """
    The agent collects shapes from dispensers, paints them with color tools,
    and delivers them to the delivery zone across a sequence of rounds.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        resolution: int = 512,
        render_action: bool = False,
        rounds_path: str | None = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.render_size = resolution
        self.window_size = 512
        self.ui_top_height = 50
        self.ui_bottom_height = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.rounds_path = rounds_path or os.path.join(current_dir, "rounds.json")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8),
            "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32),
        })

        self.variation_space = swm.spaces.Dict({
            "player": swm.spaces.Dict({
                "speed": swm.spaces.Box(low=100.0, high=400.0, init_value=200.0, shape=(), dtype=np.float32),
                "start_position": swm.spaces.Box(
                    low=50.0, high=450.0,
                    init_value=np.array([256.0, 256.0], dtype=np.float32),
                    shape=(2,), dtype=np.float32,
                ),
                "color": swm.spaces.RGBBox(init_value=np.array([65, 105, 225], dtype=np.uint8)),
            }),
            "background": swm.spaces.Dict({
                "color": swm.spaces.RGBBox(init_value=np.array([232, 189, 137], dtype=np.uint8)),
            }),
            "rewards": swm.spaces.Dict({
                "substep_reward": swm.spaces.Box(low=0.0, high=1.0, init_value=0.1, shape=(), dtype=np.float32),
                "round_complete_bonus": swm.spaces.Box(low=0.0, high=10.0, init_value=1.0, shape=(), dtype=np.float32),
                "timeout_penalty": swm.spaces.Box(low=-10.0, high=0.0, init_value=-1.0, shape=(), dtype=np.float32),
            }),
        })

        self.window = None
        self.clock = None
        self.canvas = None
        self.space = None
        self.player = None
        self.objects = {}
        self.unlocked_dispensers = set()
        self.unlocked_tools = set()
        self.collision_handler = None
        self.round_manager = None
        self.step_count = 0
        self.round_time_remaining = 0
        self.pending_reward = 0.0
        self.reward_config = {}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)
            self.variation_space.reset()

        options = options or {}

        self.reward_config = {
            "substep_reward": float(self.variation_space["rewards"]["substep_reward"].value),
            "round_complete_bonus": float(self.variation_space["rewards"]["round_complete_bonus"].value),
            "timeout_penalty": float(self.variation_space["rewards"]["timeout_penalty"].value),
        }

        self._setup_world()

        self.round_manager = RoundManager.load_from_file(self.rounds_path)
        start_round = options.get("start_round", 0)
        self.round_manager.reset(start_round)

        self._unlock_items_up_to_round(start_round)
        self._update_enabled_items()
        self._load_current_round()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        self.player.apply_action(action)
        self.space.step(PhysicsConfig.TIMESTEP)

        self.round_time_remaining -= 1
        self.step_count += 1

        reward = self.pending_reward
        self.pending_reward = 0.0
        terminated = False
        truncated = False

        dz = self.objects["delivery_zone"]
        if dz and dz.is_requirement_met():
            reward += self.reward_config["round_complete_bonus"]
            self.round_manager.advance_round()
            if self.round_manager.is_complete():
                terminated = True
            else:
                round_config = self.round_manager.get_current_round()
                self._update_unlocked_items(round_config)
                self._update_enabled_items()
                self._load_current_round()
        elif self.round_time_remaining <= 0:
            reward += self.reward_config["timeout_penalty"]
            terminated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode is None:
            return None

        if self.canvas is None:
            pygame.init()
            self.canvas = pygame.Surface((self.window_size, self.window_size))
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))

        bg_color = tuple(self.variation_space["background"]["color"].value)
        self.canvas.fill(bg_color)

        play_rect = pygame.Rect(0, self.ui_top_height, self.window_size, self.window_size - self.ui_top_height - self.ui_bottom_height)
        play_color = tuple(max(0, c - 20) for c in bg_color)
        pygame.draw.rect(self.canvas, play_color, play_rect)

        for d in self.objects["dispensers"]:
            if d.shape_type in self.unlocked_dispensers:
                draw_dispenser(self.canvas, d)

        for t in self.objects["tools"]:
            if t.target_color in self.unlocked_tools:
                draw_tool(self.canvas, t)

        dz = self.objects.get("delivery_zone")
        if dz:
            rect = pygame.Rect(dz.position[0] - dz.size[0] // 2, dz.position[1] - dz.size[1] // 2, dz.size[0], dz.size[1])
            pygame.draw.rect(self.canvas, dz.color, rect)
            pygame.draw.rect(self.canvas, (200, 255, 200), rect, 3)

        draw_player(self.canvas, self.player)

        round_config = self.round_manager.get_current_round()
        time_limit = round_config["time_limit"] if round_config else 1
        req = dz.required_item if dz else None
        completed = dz.completed if dz else False

        draw_ui(self.canvas, self.round_time_remaining, time_limit, req, completed, self.window_size, self.window_size - self.ui_bottom_height - self.ui_top_height, self.ui_bottom_height)

        if self.render_mode == "human":
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None

    def _setup_world(self):
        self.space = setup_physics_space()

        play_bottom = self.window_size - self.ui_bottom_height
        add_walls(self.space, self.window_size, self.window_size, wall_thickness=1.0,
                  top_limit=self.ui_top_height, bottom_limit=play_bottom)

        layout_config = {
            "dispensers": [
                {"position": [80, 80], "shape_type": "CIRCLE"},
                {"position": [160, 80], "shape_type": "SQUARE"},
                {"position": [240, 80], "shape_type": "TRIANGLE"},
                {"position": [320, 80], "shape_type": "STAR"},
                {"position": [400, 80], "shape_type": "HEART"},
            ],
            "tools": [
                {"position": [80, 470], "color": "RED"},
                {"position": [160, 470], "color": "BLUE"},
                {"position": [240, 470], "color": "GREEN"},
                {"position": [320, 470], "color": "YELLOW"},
                {"position": [400, 470], "color": "PURPLE"},
            ],
            "delivery_zone": {"position": [450, 275], "size": [80, 80]},
        }

        self.objects = create_default_layout(self.space, 32.0, layout_config)

        for d in self.objects["dispensers"]:
            d.disable()
        for t in self.objects["tools"]:
            t.disable()

        start_pos = self.variation_space["player"]["start_position"].value
        self.player = Player(
            self.space,
            position=tuple(start_pos),
            radius=15.0,
            max_velocity=float(self.variation_space["player"]["speed"].value),
            color=tuple(self.variation_space["player"]["color"].value),
        )

        self.collision_handler = CollisionHandler(self)
        self.collision_handler.setup_handlers(self.space)

    def _load_current_round(self):
        round_config = self.round_manager.get_current_round()
        if not round_config:
            return
        self.player.held_shape = None
        self.round_time_remaining = round_config["time_limit"]
        self.objects["delivery_zone"].set_requirement(round_config["required_item"])

    def _get_obs(self):
        img = self.render()
        proprio = np.zeros(11, dtype=np.float32)
        proprio[0] = self.player.body.position.x
        proprio[1] = self.player.body.position.y
        proprio[2] = self.player.body.velocity.x
        proprio[3] = self.player.body.velocity.y
        if self.player.held_shape:
            proprio[4] = 1.0
            proprio[5] = float(self.player.held_shape.type.value)
            proprio[6] = float(list(Color).index(self.player.held_shape.color) + 1)
        dz = self.objects["delivery_zone"]
        if dz and dz.required_item:
            req = dz.required_item
            proprio[7] = float(ShapeType[req["type"]].value)
            proprio[8] = float(list(Color).index(Color[req["color"]]) + 1)
            proprio[9] = 1.0 if dz.completed else 0.0
        round_config = self.round_manager.get_current_round()
        t_lim = round_config["time_limit"] if round_config else 1
        proprio[10] = self.round_time_remaining / t_lim
        return {"image": img, "proprio": proprio}

    def _get_info(self):
        dz = self.objects.get("delivery_zone")
        return {
            "round_index": self.round_manager.current_round_index,
            "steps": self.step_count,
            "goal": self.render(),
            "pos_agent": np.array(self.player.body.position, dtype=np.float32),
            "_required_item": dz.required_item if dz else None,
            "_completed": dz.completed if dz else False,
        }

    def _get_round_requirements(self, round_config):
        shapes_needed = set()
        colors_needed = set()
        req = round_config.get("required_item")
        if req:
            try:
                shapes_needed.add(ShapeType[req["type"]])
            except KeyError:
                pass
            try:
                colors_needed.add(Color[req["color"]])
            except KeyError:
                pass
        return shapes_needed, colors_needed

    def _update_unlocked_items(self, round_config):
        shapes, colors = self._get_round_requirements(round_config)
        self.unlocked_dispensers.update(shapes)
        self.unlocked_tools.update(colors)

    def _unlock_items_up_to_round(self, round_index):
        self.unlocked_dispensers = set()
        self.unlocked_tools = set()
        for i in range(min(round_index + 1, len(self.round_manager.rounds))):
            self._update_unlocked_items(self.round_manager.rounds[i])

    def _update_enabled_items(self):
        for d in self.objects["dispensers"]:
            if d.shape_type in self.unlocked_dispensers:
                d.enable()
            else:
                d.disable()
        for t in self.objects["tools"]:
            if t.target_color in self.unlocked_tools:
                t.enable()
            else:
                t.disable()
