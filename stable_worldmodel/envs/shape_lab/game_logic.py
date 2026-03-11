import json
import math

import pygame
import pymunk

from .entities import (
    Color,
    ColorTool,
    DeliveryZone,
    Dispenser,
    PhysicsConfig,
    Player,
    Shape,
    ShapeType,
)


def setup_physics_space() -> pymunk.Space:
    space = pymunk.Space()
    space.gravity = PhysicsConfig.GRAVITY
    space.damping = PhysicsConfig.DAMPING
    space.iterations = PhysicsConfig.ITERATIONS
    return space


def add_walls(
    space: pymunk.Space,
    map_width: float,
    map_height: float,
    wall_thickness: float = 10.0,
    top_limit: float = 0.0,
    bottom_limit: float = 0.0,
):
    effective_bottom = bottom_limit if bottom_limit > 0 else map_height
    effective_top = top_limit
    walls = [
        pymunk.Segment(space.static_body, (0, effective_top), (map_width, effective_top), wall_thickness),
        pymunk.Segment(space.static_body, (0, effective_bottom), (map_width, effective_bottom), wall_thickness),
        pymunk.Segment(space.static_body, (0, effective_top), (0, effective_bottom), wall_thickness),
        pymunk.Segment(space.static_body, (map_width, effective_top), (map_width, effective_bottom), wall_thickness),
    ]
    for wall in walls:
        wall.friction = 1.0
        wall.elasticity = 0.0
        wall.collision_type = PhysicsConfig.LAYER_WALL
    space.add(*walls)
    return walls


class CollisionHandler:
    def __init__(self, env):
        self.env = env

    def setup_handlers(self, space: pymunk.Space):
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_PLAYER,
            collision_type_b=PhysicsConfig.LAYER_DISPENSER,
            begin=self._on_player_dispenser_begin,
        )
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_PLAYER,
            collision_type_b=PhysicsConfig.LAYER_TOOL,
            begin=self._on_player_tool_begin,
        )
        space.on_collision(
            collision_type_a=PhysicsConfig.LAYER_PLAYER,
            collision_type_b=PhysicsConfig.LAYER_DELIVERY,
            begin=self._on_player_delivery_begin,
        )

    def _on_player_dispenser_begin(self, arbiter, space, data):
        player = getattr(arbiter.shapes[0], "player_obj", None)
        dispenser = getattr(arbiter.shapes[1], "dispenser_obj", None)
        if player and dispenser:
            player.held_shape = Shape(type=dispenser.shape_type, color=Color.RED)
        return True

    def _on_player_tool_begin(self, arbiter, space, data):
        player = getattr(arbiter.shapes[0], "player_obj", None)
        tool = getattr(arbiter.shapes[1], "tool_obj", None)
        if player and tool and player.held_shape:
            player.held_shape.color = tool.target_color
        return False

    def _on_player_delivery_begin(self, arbiter, space, data):
        player = getattr(arbiter.shapes[0], "player_obj", None)
        delivery_zone = getattr(arbiter.shapes[1], "delivery_obj", None)
        if player and delivery_zone and player.held_shape:
            req = delivery_zone.required_item
            if req and not delivery_zone.completed:
                try:
                    target_type = ShapeType[req["type"]]
                    target_color = Color[req["color"]]
                except KeyError:
                    return False
                if player.held_shape.matches(target_type, target_color):
                    delivery_zone.completed = True
                    player.held_shape = None
                    self.env.pending_reward += self.env.reward_config["substep_reward"]
        return False


def draw_star(surface, color, center, radius):
    points = []
    for i in range(10):
        angle = i * 36 * math.pi / 180 - math.pi / 2
        r = radius if i % 2 == 0 else radius * 0.4
        points.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
    pygame.draw.polygon(surface, color, points)


def draw_heart(surface, color, center, radius):
    x, y = center
    r = radius
    points = [
        (x, y + r * 0.8),
        (x - r, y - r * 0.4),
        (x - r * 0.5, y - r),
        (x, y - r * 0.5),
        (x + r * 0.5, y - r),
        (x + r, y - r * 0.4),
    ]
    pygame.draw.polygon(surface, color, points)


def draw_shape(surface, shape_type: ShapeType, color_val: tuple, center: tuple, radius: float):
    if shape_type == ShapeType.CIRCLE:
        pygame.draw.circle(surface, color_val, center, radius)
    elif shape_type == ShapeType.SQUARE:
        rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        pygame.draw.rect(surface, color_val, rect)
    elif shape_type == ShapeType.TRIANGLE:
        points = [
            (center[0], center[1] - radius),
            (center[0] - radius, center[1] + radius),
            (center[0] + radius, center[1] + radius),
        ]
        pygame.draw.polygon(surface, color_val, points)
    elif shape_type == ShapeType.STAR:
        draw_star(surface, color_val, center, radius)
    elif shape_type == ShapeType.HEART:
        draw_heart(surface, color_val, center, radius)


def draw_player(canvas: pygame.Surface, player: Player):
    pos = (int(player.body.position.x), int(player.body.position.y))
    pygame.draw.circle(canvas, player.color, pos, int(player.radius))
    if player.held_shape:
        draw_shape(canvas, player.held_shape.type, player.held_shape.color.value, pos, player.radius * 0.6)


def draw_dispenser(canvas: pygame.Surface, dispenser: Dispenser):
    pos = (int(dispenser.position[0]), int(dispenser.position[1]))
    size = int(dispenser.size)
    rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
    pygame.draw.rect(canvas, dispenser.color, rect)
    pygame.draw.rect(canvas, (200, 200, 200), rect, 2)
    draw_shape(canvas, dispenser.shape_type, Color.RED.value, pos, size * 0.3)


def draw_tool(canvas: pygame.Surface, tool: ColorTool):
    pos = (int(tool.position[0]), int(tool.position[1]))
    size = int(tool.size)
    rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
    pygame.draw.rect(canvas, tool.target_color.value, rect)
    pygame.draw.rect(canvas, (255, 255, 255), rect, 2)


def draw_ui(canvas, time_remaining, time_limit, requirement, completed, width, height, ui_height):
    font = pygame.font.SysFont(None, 24)
    item_size = 36
    item_x, item_y = 25, 25

    if requirement:
        try:
            stype = ShapeType[requirement["type"]]
            scolor = Color[requirement["color"]]
            border_color = (0, 255, 0) if completed else (100, 100, 100)
            pygame.draw.rect(canvas, border_color, (item_x - item_size // 2, item_y - item_size // 2, item_size, item_size), 2)
            draw_shape(canvas, stype, scolor.value, (item_x, item_y), 14)
        except Exception:
            pass

    bar_x = item_x + item_size // 2 + 15
    bar_width = width - bar_x - 20
    bar_y, bar_height = 15, 20

    pygame.draw.rect(canvas, (100, 100, 100), (bar_x, bar_y, bar_width, bar_height))

    progress = time_remaining / time_limit if time_limit > 0 else 0
    progress_width = int(bar_width * progress)
    if progress > 0.5:
        bar_color = (100, 255, 100)
    elif progress > 0.25:
        bar_color = (255, 200, 100)
    else:
        bar_color = (255, 100, 100)
    pygame.draw.rect(canvas, bar_color, (bar_x, bar_y, progress_width, bar_height))
    pygame.draw.rect(canvas, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height), 2)

    text_surf = font.render(str(time_remaining), True, (255, 255, 255))
    canvas.blit(text_surf, text_surf.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2)))


class RoundManager:
    def __init__(self, rounds_config):
        self.rounds = rounds_config
        self.current_round_index = 0

    def get_current_round(self):
        if self.current_round_index < len(self.rounds):
            return self.rounds[self.current_round_index]
        return None

    def advance_round(self):
        self.current_round_index += 1

    def reset(self, start_round=0):
        self.current_round_index = start_round

    def is_complete(self):
        return self.current_round_index >= len(self.rounds)

    @staticmethod
    def load_from_file(filepath):
        with open(filepath) as f:
            data = json.load(f)
        return RoundManager(data.get("rounds", []))


def create_default_layout(space, tile_size, layout_config):
    objects = {"dispensers": [], "tools": [], "delivery_zone": None}

    for d_conf in layout_config["dispensers"]:
        stype = ShapeType[d_conf["shape_type"]]
        objects["dispensers"].append(Dispenser(space, tuple(d_conf["position"]), stype, tile_size))

    for t_conf in layout_config["tools"]:
        c = Color[t_conf["color"]]
        objects["tools"].append(ColorTool(space, tuple(t_conf["position"]), c, tile_size))

    if "delivery_zone" in layout_config:
        dz_conf = layout_config["delivery_zone"]
        objects["delivery_zone"] = DeliveryZone(
            space, tuple(dz_conf["position"]), tuple(dz_conf["size"]), tile_size
        )

    return objects
