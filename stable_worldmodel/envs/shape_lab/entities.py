from dataclasses import dataclass
from enum import Enum

import numpy as np
import pymunk


class PhysicsConfig:
    LAYER_PLAYER = 1
    LAYER_SHAPE = 2
    LAYER_TOOL = 3
    LAYER_WALL = 4
    LAYER_DISPENSER = 5
    LAYER_DELIVERY = 6

    GRAVITY = (0, 0)
    DAMPING = 0.8
    ITERATIONS = 20
    TIMESTEP = 1 / 60


class ShapeType(Enum):
    CIRCLE = 1
    SQUARE = 2
    TRIANGLE = 3
    STAR = 4
    HEART = 5


class Color(Enum):
    RED = (231, 76, 60)
    BLUE = (52, 152, 219)
    GREEN = (46, 204, 113)
    YELLOW = (241, 196, 15)
    PURPLE = (155, 89, 182)
    ORANGE = (230, 126, 34)
    CYAN = (26, 188, 156)
    WHITE = (236, 240, 241)

    @property
    def rgb(self):
        return self.value


@dataclass
class Shape:
    type: ShapeType
    color: Color = Color.RED

    def matches(self, required_type: ShapeType, required_color: Color) -> bool:
        return self.type == required_type and self.color == required_color


class Player:
    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        tile_size: float = 32.0,
        radius: float = 16.0,
        mass: float = 1.0,
        friction: float = 0.3,
        elasticity: float = 0.0,
        max_velocity: float = 200.0,
        force_scale: float = 50.0,
        color: tuple[int, int, int] = (65, 105, 225),
    ):
        self.space = space
        self.tile_size = tile_size
        self.color = color
        self.radius = radius
        self.max_velocity = max_velocity
        self.force_scale = force_scale
        self.held_shape: Shape | None = None

        self.body = pymunk.Body(mass, float("inf"))
        self.body.position = position

        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.friction = friction
        self.shape.elasticity = elasticity
        self.shape.collision_type = PhysicsConfig.LAYER_PLAYER
        self.shape.player_obj = self

        space.add(self.body, self.shape)

    def apply_action(self, action: np.ndarray):
        direction = np.clip(action, -1.0, 1.0)
        magnitude = np.linalg.norm(direction)
        if magnitude > 1.0:
            direction = direction / magnitude

        target_velocity = direction * self.max_velocity
        current_velocity = self.body.velocity
        velocity_diff = (
            target_velocity[0] - current_velocity.x,
            target_velocity[1] - current_velocity.y,
        )
        force = (
            self.body.mass * self.force_scale * velocity_diff[0],
            self.body.mass * self.force_scale * velocity_diff[1],
        )
        self.body.apply_force_at_local_point(force, (0, 0))


class Dispenser:
    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        shape_type: ShapeType,
        tile_size: float = 32.0,
        size: float = 32.0,
    ):
        self.space = space
        self.position = position
        self.shape_type = shape_type
        self.tile_size = tile_size
        self.size = size
        self.color = (100, 100, 100)

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.shape = pymunk.Poly.create_box(self.body, (size, size), radius=2)
        self.shape.collision_type = PhysicsConfig.LAYER_DISPENSER
        self.shape.friction = 0.5
        self.shape.elasticity = 0.0
        self.shape.dispenser_obj = self

        space.add(self.body, self.shape)

    def enable(self):
        if self.body not in self.space.bodies:
            self.space.add(self.body, self.shape)

    def disable(self):
        if self.body in self.space.bodies:
            self.space.remove(self.body, self.shape)


class ColorTool:
    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        target_color: Color,
        tile_size: float = 32.0,
        size: float = 32.0,
    ):
        self.space = space
        self.position = position
        self.target_color = target_color
        self.tile_size = tile_size
        self.size = size

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.shape = pymunk.Poly.create_box(self.body, (size, size), radius=2)
        self.shape.collision_type = PhysicsConfig.LAYER_TOOL
        self.shape.sensor = True
        self.shape.tool_obj = self

        space.add(self.body, self.shape)

    def enable(self):
        if self.body not in self.space.bodies:
            self.space.add(self.body, self.shape)

    def disable(self):
        if self.body in self.space.bodies:
            self.space.remove(self.body, self.shape)


class DeliveryZone:
    def __init__(
        self,
        space: pymunk.Space,
        position: tuple[float, float],
        size: tuple[float, float],
        tile_size: float = 32.0,
    ):
        self.space = space
        self.position = position
        self.size = size
        self.tile_size = tile_size
        self.color = (50, 200, 50)

        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position
        self.shape = pymunk.Poly.create_box(self.body, size, radius=2)
        self.shape.collision_type = PhysicsConfig.LAYER_DELIVERY
        self.shape.sensor = True
        self.shape.delivery_obj = self

        space.add(self.body, self.shape)

        self.required_item = None
        self.completed = False

    def set_requirement(self, requirement: dict):
        self.required_item = requirement
        self.completed = False

    def is_requirement_met(self) -> bool:
        return self.completed
