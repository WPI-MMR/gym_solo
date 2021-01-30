from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from gym_solo.envs.solo8v2vanilla import Solo8VanillaConfig, Solo8VanillaEnv
from gym_solo import solo_types


@dataclass
class RealtimeSolo8VanillaConfig(Solo8VanillaConfig):
  dt: float = None


class RealtimeSolo8VanillaEnv(Solo8VanillaEnv):
  """A realtime solo8 vanilla gym environment.
  
  Note that the model corresponds to the solo8v2. This environment differs from
  a typical Gym environment as the concept of a "step" is relevant in this
  case. In this realtime implementation, the simulation is always running and
  the `step()` function is the same as sending position controls to the 
  robot.
  """
  def __init__(self, use_gui: bool = True, config=None, **kwargs):
    """Create a solo8 env"""
    config = config or RealtimeSolo8VanillaConfig()

    if config.dt is not None:
      raise ValueError('Cannot have a dt in a realtime simulation')

    super().__init__(use_gui, False, config)