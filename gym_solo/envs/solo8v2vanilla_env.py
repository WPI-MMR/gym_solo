import gym
import numpy as np
import pybullet
import random

from typing import Any, Dict, List, Tuple

from gym_solo import types

class Solo8VanillaEnv(gym.Env):
  def __init__(self, **kwargs) -> None:
    """Create a solo8 env"""
    pass

  def step(self, action: List[float]) -> Tuple[types.obs, float, bool, 
                                               Dict[Any, Any]]:
    """The agent takes a step in the environment.

    Args:
      action (List[float]): The torques applied to the motors in N•m. Note
        len(action) == the # of actuator

    Returns:
      Tuple[types.obs, float, bool, Dict[Any, Any]]: A tuple of the next
        observation, the reward for that step, whether or not the episode 
        terminates, and an info dict for misc diagnostic details.
    """
    pass

  def reset(self) -> types.obs:
    """Reset the state of the environment and returns an initial observation.
    
    Returns:
      types.obs: The initial observation of the space.
    """
    pass

  def render(self, mode: str = 'human', close: bool = False) -> None:
    """Initialize the rendering engine.

    Args:
      mode (str, optional): Unused--here only for the API. Defaults to 
        'human'.
      close (bool, optional): Unused--here only for the API. Defaults to 
        False.
    """
    pass

  def close(self) -> None:
    """Soft shutdown the environment. """
    pass

  def seed(self, seed: int) -> None:
    """Set the seeds for random and numpy

    Args:
      seed (int): The seed to set
    """
    np.random.seed(seed)
    random.seed(seed)