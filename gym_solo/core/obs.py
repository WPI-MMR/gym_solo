from abc import ABC, abstractmethod

from typing import List, Tuple

import pybullet as p
import numpy as np
import math

import gym
from gym import spaces

from gym_solo import solo_types


class Observation(ABC):
  @abstractmethod
  def __init__(self, body_id: int):
    pass
  
  @property
  @abstractmethod
  def observation_space(self):
    pass

  @property
  @abstractmethod
  def labels(self) -> List[str]:
    pass

  @abstractmethod
  def compute(self) -> solo_types.obs:
    pass


class ObservationFactory:
  def __init__(self):
    self._observations = []
    self._obs_space = None

  def register_observation(self, obs: Observation):
    self._observations.append(obs)

  def get_obs(self) -> Tuple[List[float], List[str]]:
    all_obs = [] 
    all_labels = []
    
    for obs in self._observations:
      all_obs.append(obs.compute())
      all_labels.append(obs.labels)

    observations = [o for obs in all_obs for o in obs]
    labels = [l for lbl in all_labels for l in lbl]

    return observations, labels

  def get_observation_space(self, generate=False) -> spaces.Box:
    if self._obs_space and not generate:
      return self._obs_space

    lower, upper = [], []
    for obs in self._observations:
      lower.extend(obs.observation_space.low)
      upper.extend(obs.observation_space.high)

    self._obs_space = spaces.Box(low=lower, high=upper)
    return self._obs_space


class MotorEncoders(Observation):
  pass


class TorsoIMU(Observation):
  labels: List[str] = ['θ_x', 'θ_y', 'θ_z', 'vx', 'vy', 'vz', 'wx', 'wy', 'z']

  def __init__(self, body_id: int, degrees=False):
    self.robot = body_id
    self._degrees = degrees

  @property
  def observation_space(self):
    angle_min = -180. if self.degrees else -math.pi
    angle_max = 180 if self.degrees else math.pi

    lower = [angle_min, angle_min, angle_min, # Orientation
             -np.inf, -np.inf, -np.inf,       # Linear Velocity
             -np.inf, -np.inf, -np.inf]       # Angular Velocity
    upper = [angle_max, angle_max, angle_max, # Same as above
             np.inf, np.inf, np.inf,          
             np.inf, np.inf, np.inf]         

    return spaces.Box(low=lower, high=upper)

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