from abc import ABC, abstractmethod

import pybullet as p
import numpy as np

from gym_solo import solo_types


class Observation(ABC):
  @abstractmethod
  def __init__(self, body_id: int):
    pass
  
  @property
  @abstractmethod
  def observation_space(self):
    pass

  @abstractmethod
  def compute(self) -> solo_types.obs:
    pass


class ObservationFactory:
  def __init__(self):
    self._observations = []

  def register_observation(self, obs: Observation):
    self._observations.append(obs)

  def get_obs(self):
    return np.array([obs.compute() for obs in self._observations]).flatten()


class Test(Observation):
  def __init__(self, body_id: int):
    pass
  
  def observation_space(self):
    pass

  def compute(self) -> solo_types.obs:
    return [2.3, 4.5]

class MotorEncoders(Observation):
  pass


class TorsoIMU(Observation):
  pass


class FootDistances(Observation):
  pass