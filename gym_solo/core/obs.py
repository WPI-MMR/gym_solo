from abc import ABC, abstractmethod

from pybullet_utils import bullet_client
from typing import List, Tuple

import pybullet as p
import numpy as np
import math

import gym
from gym import spaces

from gym_solo import solo_types


class Observation(ABC):
  """An observation for a body in the pybullet simulation.

  Attributes:
    _client: The PyBullet client for the instance. Will be set via a
      property setter.
  """
  _client: bullet_client.BulletClient = None

  @abstractmethod
  def __init__(self, body_id: int):
    """Create a new Observation.
    
    Note that for every child of this class, each one *needs* to specify
    which pybullet body id they would like to track the observation for.

    Args:
      body_id (int): PyBullet body id to get the observation for.
    """
    pass
  
  @property
  @abstractmethod
  def observation_space(self) -> spaces.Space:
    """Get the observation space of the Observation.

    Returns:
      spaces.Space: The observation space.
    """
    pass

  @property
  @abstractmethod
  def labels(self) -> List[str]:
    """A list of labels corresponding to the observation.
    
    i.e. if the observation was [1, 2, 3], and the labels were ['a', 'b', 'c'],
    then a = 1, b = 2, c = 3.

    Returns:
      List[str]: Labels, where the index is the same as its respective 
      observation.
    """
    pass

  @abstractmethod
  def compute(self) -> solo_types.obs:
    """Compute the observation for the current state.

    Returns:
      solo_types.obs: Specified observation for the current state.
    """
    pass

  @property
  def client(self) -> bullet_client.BulletClient:
    """Get the Observation's physics client.

    Raises:
      ValueError: If the PyBullet client hasn't been set yet.

    Returns:
      bullet_client.BulletClient: The active client for the observation.
    """
    if not self._client:
      raise ValueError('PyBullet client needs to be set')
    return self._client

  @client.setter
  def client(self, client: bullet_client.BulletClient):
    """Set the Observation's physics client.

    Args:
      client (bullet_client.BulletClient): The client to use for the 
        observation.
    """
    self._client = client


class ObservationFactory:
  def __init__(self, client: bullet_client.BulletClient, 
               normalize: bool = False):
    """Create a new Observation Factory.

    Args:
      client (bullet_client.BulletClient): Pybullet client to perform 
        calculations.
    """
    self._client = client
    self._observations = []
    self._obs_space = None
    self._normalize = normalize

  def register_observation(self, obs: Observation):
    """Add an observation to be computed.

    Args:
      obs (Observation): Observation to be tracked.
    """
    obs.client = self._client

    lbl_len = len(obs.labels)
    obs_space_len = len(obs.observation_space.low)
    obs_len = obs.compute().size

    if lbl_len != obs_space_len:
      raise ValueError('Labels have length {} != obs space len {}'.format(
        lbl_len, obs_space_len))
    if lbl_len != obs_len:
      raise ValueError('Labels have length {} != obs len {}'.format(
        lbl_len, obs_len))
    
    self._observations.append(obs)

  def get_obs(self) -> Tuple[solo_types.obs, List[str]]:
    """Get all of the observations for the current state.

    Returns:
      Tuple[solo_types.obs, List[str]]: The observations and associated labels.
        len(observations) == len(labels) and labels[i] corresponds to the
        i-th observation.
    """
    if not self._observations:
      raise ValueError('Need to register at least one observation instance')

    all_obs = [] 
    all_labels = []
    
    for obs in self._observations:
      all_labels.append(obs.labels)

      values = obs.compute()
      if self._normalize:
        a = np.array(values)
        low = obs.observation_space.low
        hi = obs.observation_space.high
        values = (2 * (a - low)) / (hi - low) - 1

      all_obs.append(values)

    observations = np.concatenate(all_obs)
    labels = [l for lbl in all_labels for l in lbl]

    return observations, labels

  def get_observation_space(self, generate=False) -> spaces.Box:
    """Get the combined observation space of all of the registered observations.

    Args:
      generate (bool, optional): Whether or not to regenerate the observation
        space or just used the cached ersion. Note that some Observations
        might dynamically generate their observation space, so this could be a
        potentially expensive operation. Defaults to False.

    Raises:
      ValueError: If no observations are registered.

    Returns:
      spaces.Box: The observation space of the registered Observations.
    """
    if not self._observations:
      raise ValueError('Can\'t generate an empty observation space')
    if self._obs_space and not generate:
      return self._obs_space

    lower, upper = [], []
    for obs in self._observations:
      lower.extend(obs.observation_space.low)
      upper.extend(obs.observation_space.high)

    if self._normalize:
      self._obs_space = spaces.Box(low=-1, high=1, shape=(len(lower),))
    else:
      self._obs_space = spaces.Box(low=np.array(lower), high=np.array(upper))
    return self._obs_space


class TorsoIMU(Observation):
  """Get the orientation and velocities of the Solo 8 torso.

  Attributes:
    labels (List[str]): The labels associated with the outputted observation
    robot (int): PyBullet BodyId for the robot.
  """
  # TODO: Add angular acceleration to support production IMUs
  labels: List[str] = ['θx', 'θy', 'θz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']

  def __init__(self, body_id: int, degrees: bool = False, 
               max_lin_velocity: float = 15, 
               max_angular_velocity: float = 10.):
    """Create a new TorsoIMU observation

    Args:
      body_id (int): The PyBullet body id for the robot.
      degrees (bool, optional): Whether or not to return angles in degrees. 
        Defaults to False.
      max_lin_velocity (float, optional): The maximum linear velocity read by
        TorsoIMU. Any torso reading made past this will just get clamped. The
        unit is arbritary, but it is recommended that this is experimentally
        found. Defaults to 10.
      max_ang_velocity (float, optional): The maximum angular velocity read by
        TorsoIMU. Any torso reading made past this will just get clamped. The
        unit is arbritary, but it is recommended that this is experimentally
        found. Units are in rad/s unless degrees == True. Deaults to 10.
    """
    self.robot = body_id
    self._degrees = degrees
    self._max_lin = max_lin_velocity
    self._max_ang = max_angular_velocity

    self._low = None
    self._high = None
    self.observation_space # Populate the bounds incase it dosn't get called

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

    lower = [angle_min, angle_min, angle_min,                 # Orientation
             -self._max_lin, -self._max_lin, -self._max_lin,  # Linear Velocity
             -self._max_ang, -self._max_ang, -self._max_ang]  # Angular Velocity

    upper = [angle_max, angle_max, angle_max,                 # Same as above
             self._max_lin, self._max_lin, self._max_lin,          
             self._max_ang, self._max_ang, self._max_ang]         

    if not (self._low and self._high):
      self._low = lower
      self._high = upper

    return spaces.Box(low=np.array(lower), high=np.array(upper))

  def compute(self) -> solo_types.obs:
    """Compute the torso IMU values for a state.

    Returns:
      solo_types.obs: The observation for the current state (accessed via
        pybullet). Note the values are bounded by self.observation_space even
        if the true value is greater than that (i.e. the observation is clipped)
    """
    _, orien_quat = self.client.getBasePositionAndOrientation(self.robot)

    # Orien is in (x, y, z)
    orien = np.array(self.client.getEulerFromQuaternion(orien_quat))

    v_lin, v_ang = self.client.getBaseVelocity(self.robot)
    v_lin = np.array(v_lin) 
    v_ang = np.array(v_ang)

    if self._degrees:
      orien = np.degrees(orien)
      v_ang = np.degrees(v_ang)

    raw_values = np.concatenate([orien, v_lin, v_ang])
    return np.clip(raw_values, self._low, self._high)


class MotorEncoder(Observation):
  """Get the position of the all the joints
  """

  def __init__(self, body_id: int, degrees: bool = False, 
               max_rotation: float = None):
    """Create a new MotorEncoder observation

    Args:
      body_id (int): The PyBullet body id for the robot.
      degrees (bool, optional): Whether or not to return angles in degrees. 
        Defaults to False.
      max_rotation (float, optional): Artificially limit the range of the 
        motor encoders. Note then that the motor encoder observation space
        then becomes (low=[-max_rotation] * joints, high=[max_rotation] * joints).
        Defaults to the max values as per the URDF.
    """
    self.robot = body_id
    self._degrees = degrees
    self._max_rot = max_rotation

  @property
  def _num_joints(self):
    return self.client.getNumJoints(self.robot)
  
  @property
  def observation_space(self) -> spaces.Space:
    """Gets the observation space for the joints

    Returns::
      spaces.Space: The observation space corresponding to the labels
    """
    if self._max_rot:
      return spaces.Box(low=-self._max_rot, high=self._max_rot, 
                        shape=(self._num_joints, ))

    lower, upper = [], []
    for joint in range(self._num_joints):
      joint_info = self.client.getJointInfo(self.robot, joint)
      lower.append(joint_info[8])
      upper.append(joint_info[9])

    lower = np.array(lower)
    upper = np.array(upper)

    if self._degrees:
      lower = np.degrees(lower)
      upper = np.degrees(upper)

    return spaces.Box(low=lower, high=upper)

  @property
  def labels(self) -> List[str]:
    """A list of labels corresponding to the observation.

    Returns:
      List[str]: Labels, where the index is the same as its respective 
      observation.
    """
    return [self.client.getJointInfo(self.robot, joint)[1].decode('UTF-8') 
            for joint in range(self._num_joints)]

  def compute(self) -> solo_types.obs:
    """Computes the motor position values all the joints of the robot 
    for the current state.

    Returns:
      solo_types.obs: The observation extracted from pybullet
    """
    joint_values = np.array([self.client.getJointState(self.robot, i)[0] 
                             for i in range(self._num_joints)])

    if self._degrees:
      joint_values = np.degrees(joint_values)
      
    return joint_values