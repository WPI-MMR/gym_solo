from gym_solo.core import obs
from gym_solo.core import rewards

from gym import spaces
import numpy as np


class CompliantObs(obs.Observation):
  observation_space = spaces.Box(low=np.array([0., 0.]), 
                                 high=np.array([3., 3.]))
  labels = ['1', '2']

  def __init__(self, body_id):
    pass

  def compute(self):
    return np.array([1, 2])


class SimpleReward(rewards.Reward):
  def compute(self):
    return 1


class ReflectiveReward(rewards.Reward):
  def __init__(self, return_value):
    self._return_value = return_value

  def compute(self):
    return self._return_value