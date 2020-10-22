from gym_solo import solo_types

from abc import ABC, abstractmethod
import pybullet as p
import numpy as np


class ObservationFactory:
  def __init__(self):
    self._obs_fns = []

  def register_observation(self, fn):
    self._obs_fns.append(fn)

  def get_obs(self):
    return np.array([fn() for fn in self._obs_fns]).flatten()

    
class Observation(ABC):
  def __init__(self, body_id: int):
    pass
  
  @property
  @abstractmethod
  def observation_space(self):
    pass

  @abstractmethod
  def compute(self) -> List[None]:
    pass


class MotorEncoders(Observation):
  pass


class TorsoIMU(Observation):
  pass


class FootDistances(Observation):
  pass