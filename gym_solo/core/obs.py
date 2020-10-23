from abc import ABC, abstractmethod

import pybullet as p
import numpy as np
import math

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
    all_obs = [obs.compute() for obs in self._observations]
    flat = [v for obs in all_obs for v in obs]
    return np.array(flat)


class Test(Observation):
  def __init__(self, body_id: int):
    pass
  
  @property
  def observation_space(self):
    pass

  def compute(self) -> solo_types.obs:
    return [2.3, 4.5]


class MotorEncoders(Observation):
  pass


class TorsoIMU(Observation):
  def __init__(self, body_id: int, degrees=False):
    self.robot = body_id
    self._degrees = degrees

  @property
  def observation_space(self):
    pass

  def compute(self) -> solo_types.obs:
    _, orien_quat = p.getBasePositionAndOrientation(self.robot)

    # Orien is in (x, y, z)
    orien = p.getEulerFromQuaternion(orien_quat)
    v_lin, v_ang = p.getBaseVelocity(self.robot)

    if self._degrees:
        to_degrees = lambda lst: [l * 180. / math.pi for l in lst]

        orien = to_degrees(orien)
        v_ang = to_degrees(v_ang)

    return orien + v_lin + v_ang

class FootDistances(Observation):
  pass