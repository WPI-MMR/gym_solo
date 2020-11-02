from gym_solo.core import obs
from gym_solo.core import rewards

from gym import spaces
import numpy as np

from gym_solo import solo_types


class CompliantObs(obs.Observation):
  """A simple observation which implements the Observation interface.
  
  Note that the observation always returns [1, 2].

  Attributes:
    observation_space (spaces.Space): The observation space. Always returns a 
      box with a low of 0 and a high of 3.
    labels (List[int]): The labels for the two observations. Always returns
      ['1', '2']
  """
  observation_space = spaces.Box(low=np.array([0., 0.]), 
                                 high=np.array([3., 3.]))
  labels = ['1', '2']

  def __init__(self, body_id: int):
    """Create a new CompliantObs. Only kept for API conformance."""
    pass

  def compute(self) -> solo_types.obs:
    """Compute the observation for the state.

    Returns:
      solo_types.obs: [1., 2.]
    """
    return np.array([1., 2.])


class SimpleReward(rewards.Reward):
  """A reward which will always return 1."""
  def compute(self) -> float:
    """'Compute' the reward for the step. Always returns 1.

    Returns:
      float: 1.0
    """
    return 1


class ReflectiveReward(rewards.Reward):
  """A reward which returns a configurable fixed value."""
  def __init__(self, return_value: float):
    """Create a ReflectiveReward.

    Args:
      return_value (float): What value this reward should return.
    """
    self._return_value = return_value

  def compute(self) -> float:
    """Return the fixed reward.

    Returns:
      float: the configured reward.
    """
    return self._return_value