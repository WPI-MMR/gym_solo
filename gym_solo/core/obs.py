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
  def observation_space(self) -> spaces.Space:
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
    # TODO: Assert that the observation is valid
    self._observations.append(obs)

  def get_obs(self) -> Tuple[solo_types.obs, List[str]]:
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
  """Get the orientation and velocities of the Solo 8 torso.

  Attributes:
    labels (List[str]): The labels associated with the outputted observation
    robot (int): PyBullet BodyId for the robot.
  """
  labels: List[str] = ['θx', 'θy', 'θz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']

  def __init__(self, body_id: int, degrees: bool = False):
    """Create a new TorsoIMU observation

    Args:
      body_id (int): The PyBullet body id for the robot.
      degrees (bool, optional): Whether or not to return angles in degrees. 
        Defaults to False.
    """
    self.robot = body_id
    self._degrees = degrees

  @property
  def observation_space(self) -> spaces.Box:
    """Get the observation space for the IMU mounted on the Torso. 

    The IMU in this case return the orientation of the torso (x, y, z angles),
    the linear velocity (vx, vy, vz), and the angular velocity (wx, wy, wz). 
    Note that the range for the angle measurements is [-180, 180] degrees. This
    value can be toggled between degrees and radians at instantiation.

    Returns:
      spaces.Box: The observation space corresponding to (θx, θy, θz, vx, vy, 
        vz, wx, wy, wz)
    """
    angle_min = -180. if self._degrees else -np.pi
    angle_max = 180. if self._degrees else np.pi

    lower = [angle_min, angle_min, angle_min, # Orientation
             -np.inf, -np.inf, -np.inf,       # Linear Velocity
             -np.inf, -np.inf, -np.inf]       # Angular Velocity

    upper = [angle_max, angle_max, angle_max, # Same as above
             np.inf, np.inf, np.inf,          
             np.inf, np.inf, np.inf]         

    return spaces.Box(low=np.array(lower), high=np.array(upper))

  def compute(self) -> solo_types.obs:
    """Compute the torso IMU values for a state.

    Returns:
      solo_types.obs: The observation for the current state (accessed via
        pybullet)
    """
    _, orien_quat = p.getBasePositionAndOrientation(self.robot)

    # Orien is in (x, y, z)
    orien = np.array(p.getEulerFromQuaternion(orien_quat))

    v_lin, v_ang = p.getBaseVelocity(self.robot)
    v_lin = np.array(v_lin) 
    v_ang = np.array(v_ang)

    if self._degrees:
      orien = np.degrees(orien)
    else:
      v_ang = np.radians(v_ang)

    return np.concatenate([orien, v_lin, v_ang])

class FootDistances(Observation):
  pass