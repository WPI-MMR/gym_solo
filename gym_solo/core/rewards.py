from abc import ABC, abstractmethod
from dataclasses import dataclass
from pybullet_utils import bullet_client
from typing import List

import numpy as np
import pybullet as p

from gym_solo import solo_types


class Reward(ABC):  
  """A reward for a body in a pybullet simulation.

  Attributes:
    _client: The PyBullet client for the instance. Will be set via a
      property setter.
  """
  _client: bullet_client.BulletClient = None

  @abstractmethod
  def compute(self) -> solo_types.reward:
    """Compute the reward for the current state.

    Returns:
      solo_types.reward: The reward evalulated at the current state.
    """
    pass

  @property
  def client(self) -> bullet_client.BulletClient:
    """Get the reward's physics client.

    Raises:
      ValueError: If the PyBullet client hasn't been set yet.

    Returns:
      bullet_client.BulletClient: The active client for the reward.
    """
    if self._client is None:
      raise ValueError('PyBullet client needs to be set')
    return self._client

  @client.setter
  def client(self, client: bullet_client.BulletClient):
    """Set the reward's physics client.

    Args:
      client (bullet_client.BulletClient): The client to use for the 
        reward.
    """
    self._client = client


@dataclass
class _WeightedReward:
  reward: Reward
  weight: float


class RewardFactory:
  """A factory to dynamically create rewards.
  
  Note that this factory is currently implemented to combined rewards via
  a linear combination. For example, if the user wanted to register rewards
  r1, r2, and r3, the final reward would be r1 + r2 + r3. 
  
  Obviously, you can add coefficients to the rewards, and that functionality 
  is further explained in register_reward() below. 
  
  If you need more functionality then a linear combination (exponential 
  temporal decay), then it's probably in your best interest to implement that
  in a custom Reward.
  """
  def __init__(self, client: bullet_client.BulletClient):
    """Create a new RewardFactory.
    
    Args:
      client (bullet_client.BulletClient): Sandboxed Pybullet client.
    """
    self._client = client
    self._rewards: List[_WeightedReward] = []

  def register_reward(self, weight: float, reward: Reward):
    """Register a reward to be computed per state.

    Args:
      weight (float): The weight to be applied to this reward when it is 
        is combined linearly with the other rewards. The domain for this
        value is (-∞, ∞).
      reward (Reward): A Reward object which .compute() will be called on at
        reward computation time.
    """
    reward.client = self._client
    self._rewards.append(_WeightedReward(reward=reward, weight=weight))

  def get_reward(self) -> float:
    """Evaluate the current state and get the combined reward.

    Exceptions:
      ValueError: If get_reward() is called with no registered rewards.

    Returns:
      float: The reward from the current state. Note that this reward is a 
      combination of multiple atomic sub-rewards, as explained by the 
      strategies earlier.
    """
    if not self._rewards:
      raise ValueError('Need to register at least one reward instance')

    return sum(wr.weight * wr.reward.compute() for wr in self._rewards)


class UprightReward(Reward):
  """A reward for being fully upright. Note that this is technically only
  designed for the Solo8v2Vanilla and should be extended in the future to 
  support more models.

  The reward's is in the interval [-1, 1], where 1 indicates that the torso
  is upright, while -1 means that it is upside down. Observe that this means
  that the reward is orientation-specific.
  """
  _fully_upright = -np.pi / 2

  def __init__(self, robot_id: int):
    """Create a new UprightReward.

    Args:
      robot_id ([int]): The PyBullet body id of the robot
    """
    self._robot_id = robot_id

  def compute(self) -> float:
    """Compute the UprightReward for the current state. 
    
    As this reward function is specifically designed for the solo8v2vanilla 
    model, this means that the model just needs to be rotated -pi/2 radians 
    about the y axis.

    Returns:
      float: A real-valued number in [-1, 1], where 1 means perfectly upright 
      whilst -1 means that the robot is literally upside down. 
    """
    _, quat = self.client.getBasePositionAndOrientation(self._robot_id)
    unused_x, y, unused_z = np.array(self.client.getEulerFromQuaternion(quat))
    return self._fully_upright * y / self._fully_upright ** 2


class HomePositionReward(Reward):
  """Rewards the robot for being in the home position. Currently, this rewards
  the robot for being orientated properly and being as tall as possible. This
  will require some experimentation--if the robot ends up jumping to maximize
  the reward, this might need to be modified to a more intelligent height 
  reward.
  """
  # Found via experimentation--should be close enough to prevent any 
  # non-trivial errors
  _quad_standing_height = 0.3
  _max_angle = np.pi

  def __init__(self, robot_id: int):
    """Create a new HomePositionReward.

    Args:
      robot_id (int): The PyBullet body id of the robot
    """
    self._robot_id = robot_id

  def compute(self) -> float:
    """Compute the HomePositionReward for the current state. 

    Returns:
      float: A real-valued in [0, 1], where 1 is in the home position.
    """
    (unused_x, unused_y, z), quat = self.client.getBasePositionAndOrientation(
      self._robot_id)
    theta_x, theta_y, unused_z = np.array(
      self.client.getEulerFromQuaternion(quat))

    x_reward = self._max_angle - abs(theta_x)
    y_reward = self._max_angle - abs(theta_y)

    orientation_reward = (x_reward + y_reward) / (2 * self._max_angle)
    height_reward = z / self._quad_standing_height
    
    # Currently magic numbers for the relative weighting--will probably need
    # to be tuned down the line
    return 0.25 * orientation_reward + 0.75 * height_reward