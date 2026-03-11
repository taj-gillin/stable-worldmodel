from enum import Enum, auto

import numpy as np

from .entities import Color, ShapeType


class GoalType(Enum):
    GET_SHAPE = auto()
    PAINT_SHAPE = auto()
    DELIVER_SHAPE = auto()
    IDLE = auto()


class ShapeLabExpert:
    """
    Rule-based expert that solves rounds by navigating to the required dispenser,
    color tool, and delivery zone in sequence. State is re-evaluated every step
    for automatic error recovery on accidental collisions.
    """

    ARRIVAL_THRESHOLD = 15.0
    ACTION_SMOOTHING = 0.3

    # Safe middle lane between dispenser row (y≈80) and tool row (y≈470)
    SAFE_Y = 280
    DISPENSER_ZONE_Y = 120
    TOOL_ZONE_Y = 435
    LATERAL_THRESHOLD = 30.0

    def __init__(self, env, debug: bool = False):
        self.env = env
        self.debug = debug
        self.current_goal = GoalType.IDLE
        self.target_position = None
        self._prev_action = None
        self._waypoints: list[np.ndarray] = []
        self._last_goal_key = None

    def reset(self):
        self.current_goal = GoalType.IDLE
        self.target_position = None
        self._prev_action = None
        self._waypoints = []
        self._last_goal_key = None

    def get_action(self, obs: dict, info: dict) -> np.ndarray:
        if info.get("_completed", False):
            return np.zeros(2, dtype=np.float32)

        self._update_goal(info)

        if self.target_position is None:
            return np.zeros(2, dtype=np.float32)

        player_pos = self._get_player_pos()

        while self._waypoints:
            if np.linalg.norm(self._waypoints[0] - player_pos) < self.ARRIVAL_THRESHOLD:
                self._waypoints.pop(0)
            else:
                break

        current_target = self._waypoints[0] if self._waypoints else self.target_position
        raw_action = self._navigate_to(current_target, player_pos)

        if self._prev_action is not None:
            action = (1 - self.ACTION_SMOOTHING) * raw_action + self.ACTION_SMOOTHING * self._prev_action
        else:
            action = raw_action

        self._prev_action = action.copy()
        return action

    def _update_goal(self, info: dict):
        req = info.get("_required_item")
        if not req:
            self._set_goal(GoalType.IDLE, None)
            return

        player = self.env.player

        if player.held_shape is None:
            self._set_goal(GoalType.GET_SHAPE, self._get_dispenser_pos(req["type"]))
        elif player.held_shape.type.name != req["type"]:
            if self.debug:
                print(f"[Expert] Wrong type held ({player.held_shape.type.name}), need {req['type']} → re-routing")
            self._set_goal(GoalType.GET_SHAPE, self._get_dispenser_pos(req["type"]))
        elif player.held_shape.color.name != req["color"]:
            self._set_goal(GoalType.PAINT_SHAPE, self._get_tool_pos(req["color"]))
        else:
            self._set_goal(GoalType.DELIVER_SHAPE, self._get_delivery_pos())

    def _set_goal(self, goal: GoalType, target: np.ndarray | None):
        new_key = (goal, tuple(target.tolist()) if target is not None else None)
        if new_key == self._last_goal_key:
            return

        self.current_goal = goal
        self.target_position = target
        self._last_goal_key = new_key
        self._waypoints = self._compute_waypoints(target) if target is not None else []

        if self.debug:
            print(f"[Expert] Goal: {goal.name} → target={target}")

    def _compute_waypoints(self, target: np.ndarray) -> list[np.ndarray]:
        """
        Routes through a safe middle lane (SAFE_Y) when transitioning between
        the dispenser row (top) and tool row (bottom) to avoid accidental interactions.
        """
        player_pos = self._get_player_pos()
        px, py = float(player_pos[0]), float(player_pos[1])
        tx, ty = float(target[0]), float(target[1])

        in_danger = py < self.DISPENSER_ZONE_Y or py > self.TOOL_ZONE_Y
        target_in_danger = ty < self.DISPENSER_ZONE_Y or ty > self.TOOL_ZONE_Y
        needs_lateral = abs(px - tx) > self.LATERAL_THRESHOLD

        waypoints: list[np.ndarray] = []

        if in_danger:
            waypoints.append(np.array([px, self.SAFE_Y], dtype=np.float32))

        if needs_lateral and (in_danger or target_in_danger):
            waypoints.append(np.array([tx, self.SAFE_Y], dtype=np.float32))
        elif target_in_danger and not in_danger and needs_lateral:
            waypoints.append(np.array([tx, self.SAFE_Y], dtype=np.float32))

        waypoints.append(target.copy())
        return waypoints

    def _get_player_pos(self) -> np.ndarray:
        return np.array(
            [self.env.player.body.position.x, self.env.player.body.position.y],
            dtype=np.float32,
        )

    def _get_dispenser_pos(self, shape_type_name: str) -> np.ndarray | None:
        try:
            target_type = ShapeType[shape_type_name]
        except KeyError:
            return None
        for dispenser in self.env.objects["dispensers"]:
            if dispenser.shape_type == target_type:
                return np.array(dispenser.position, dtype=np.float32)
        return None

    def _get_tool_pos(self, color_name: str) -> np.ndarray | None:
        try:
            target_color = Color[color_name]
        except KeyError:
            return None
        for tool in self.env.objects["tools"]:
            if tool.target_color == target_color:
                return np.array(tool.position, dtype=np.float32)
        return None

    def _get_delivery_pos(self) -> np.ndarray | None:
        dz = self.env.objects.get("delivery_zone")
        return np.array(dz.position, dtype=np.float32) if dz else None

    def _navigate_to(self, target: np.ndarray, player_pos: np.ndarray) -> np.ndarray:
        direction = target - player_pos
        distance = np.linalg.norm(direction)
        if distance < self.ARRIVAL_THRESHOLD:
            return np.zeros(2, dtype=np.float32)
        return np.clip(direction / distance, -1.0, 1.0).astype(np.float32)
