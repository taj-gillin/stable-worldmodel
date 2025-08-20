import os

from gymnasium.envs.registration import register

from . import collect, data, policy, solver, wrappers, wm, tmp
from .utils import create_pil_image_from_url, set_state
from .world import World, SaveInitAndGoal, PushTRolloutCompletion
from .evaluator import Evaluator
from .env_trans import BackgroundDeform, ColorDeform, ShapeDeform, ImageNetDeform

register(
    id="xenoworlds/ImagePositioning-v1",
    entry_point="xenoworlds.envs.image_positioning:ImagePositioning",
)

register(
    id="xenoworlds/PushT-v1",
    entry_point="xenoworlds.envs.pusht:PushT",
)
