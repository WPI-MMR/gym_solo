from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.testing._private.utils import requires_memory
from pybullet_utils import bullet_client
from typing import Tuple, List

import numpy as np
import pybullet as p
import math

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

    x_reward = (self._max_angle - abs(theta_x)) / (self._max_angle * 2)
    y_reward = (self._max_angle - abs(theta_y)) / (self._max_angle * 2)

    orientation_reward = x_reward + y_reward
    height_reward = z / self._quad_standing_height
    
    # Currently magic numbers for the relative weighting--will probably need
    # to be tuned down the line
    return 0.25 * orientation_reward + 0.75 * height_reward

    
class FlatTorsoReward(Reward):
  """Rewards the agent for keeping toros relatively flat."""
  def __init__(self, robot_id: int, hard_margin: float = .1, 
               soft_margin: float = 0.1):
    """Create a new FlatTorsoReward

    Args:
      robot_id (int): the client-specific pybullet id 
      hard_margin (float, optional): the margin where the perfect reward is 
        given. For example, with a target speed of 0 and a margin of 0.1, a 
        speed of 0.05 will still give a reward of 1. Defaults to .1.
      soft_margin (float, optional): how long to have a downward sloping 
        reward function. The value at the soft_margin is effectively 0. Defaults
        to 0.1.
    """
    self._robot_id = robot_id
    self._hard_margin = hard_margin
    self._soft_margin = soft_margin

  def compute(self) -> float:
    """Compute the FlatTorsoReward for the current state.

    Returns:
      float: A real value in [0, 1], 1 if the horizontal velocity is near the
        target velocity. The specific behavior of how "near" is defined is
        in the constructor.
    """
    _, quat = self.client.getBasePositionAndOrientation(self._robot_id)
    theta_x, theta_y, _ = self.client.getEulerFromQuaternion(quat)
    rmse = math.sqrt(theta_x ** 2 + theta_y ** 2)
    return tolerance(rmse, bounds=(-self._hard_margin, 
                                   self._hard_margin),
                     margin=self._soft_margin)


class SmallControlReward(Reward):
  """Rewards the robot for making minimal movements.
  
  This reward is useful for rewarding "stable" behavior down the line so that
  the robot learns to make smoother and smaller movements.
  """

  def __init__(self, robot_id: int, margin: float = 1.):
    """Create a new SmallControlReward.

    Args:
      robot_id (int): The pybullet ID of the robot
      margin (float, optional): Control the steepness of the decline of the 
        reward. Defaults to 1..
    """
    self._robot_id = robot_id
    self._margin = margin

  def compute(self) -> float:
    """Compute the SmallControlReward for the current state.

    Returns:
      float: A real value in [0, 1], where 1 is if the robot is completely 
        still.
    """
    joint_cnt = self.client.getNumJoints(self._robot_id)
    joint_velocities = np.array([self.client.getJointState(self._robot_id, i)[1]
                                 for i in range(joint_cnt)])
    avg_angular_speed = np.average(np.abs(joint_velocities))
    return tolerance(avg_angular_speed, margin=self._margin)


class HorizontalMoveSpeedReward(Reward):
  """Rewards the agent for maintaining a specific horizontal speed. """
  def __init__(self, robot_id: int, target_speed: int, hard_margin: float = .1,
               soft_margin: float = 0.1):
    """Create a new HorizontalMoveSpeedReward

    Args:
      robot_id (int): the client-specific pybullet id 
      target_speed (int): the target speed to maintain. The speed is computed
        by the magnitude of the velocity vector.
      hard_margin (float, optional): the margin where the perfect reward is 
        given. For example, with a target speed of 0 and a margin of 0.1, a 
        speed of 0.05 will still give a reward of 1. Defaults to .1.
      soft_margin (float, optional): how long to have a downward sloping 
        reward function. The value at the soft_margin is effectively 0. Defaults
        to 0.1.
    """
    self._robot_id = robot_id
    self._target_speed = target_speed
    self._hard_margin = hard_margin
    self._soft_margin = soft_margin
    
  def compute(self) -> float:
    """Compute the HorizontalMoveSpeedReward for the current state.

    Returns:
      float: A real value in [0, 1], 1 if the horizontal velocity is near the
        target velocity. The specific behavior of how "near" is defined is
        in the constructor.
    """
    (vx, vy, _), _ = self.client.getBaseVelocity(self._robot_id)
    speed = math.sqrt(vx ** 2 + vy ** 2)
    return tolerance(speed, bounds=(self._target_speed - self._hard_margin, 
                                    self._target_speed + self._hard_margin),
                     margin=self._soft_margin)


class TorsoHeightReward(Reward):
  """Rewards the robot for maintaining a certain height with its Torso. """
  def __init__(self, robot_id: int, target_height: int, hard_margin: float = .1,
               soft_margin: float = 0.1):
    """Create a new HorizontalMoveSpeedReward

    Args:
      robot_id (int): the client-specific pybullet id 
      target_height (int): the target height to maintain. 
      hard_margin (float, optional): the margin where the perfect reward is 
        given. For example, with a target speed of 0 and a margin of 0.1, a 
        speed of 0.05 will still give a reward of 1. Defaults to .1.
      soft_margin (float, optional): how long to have a downward sloping 
        reward function. The value at the soft_margin is effectively 0. Defaults
        to 0.1.
    """
    self._robot_id = robot_id
    self._target_height = target_height
    self._hard_margin = hard_margin
    self._soft_margin = soft_margin

  def compute(self) -> float:
    """Compute the TorsoHeightReward for the current state.

    Returns:
      float: A real value in [0, 1], 1 if the Torso's height is near the
        target height. The specific behavior of how "near" is defined is in the 
        constructor.
    """
    (_, _, z), _ = self.client.getBasePositionAndOrientation(self._robot_id)
    return tolerance(z, bounds=(self._target_height - self._hard_margin,
                                self._target_height + self._hard_margin),
                     margin=self._soft_margin)


def tolerance(x: float, bounds: Tuple[float, float] = (0., 0.), 
              margin: float = 0., margin_value: float = 1e-6):
  """
  Create a sloped reward function about a given bounds range.

  Args:
    x (float): The value to evaluate.
    bounds ((float, float)): A tuple of (lower, upper). If `x` falls between
      `lower` and `upper`, then the the returned value is 1. Otherwise the
      behavior is as described in the `Returns` section.
    margin (float): How steeply to decline the reward function. If `margin`
      is 0, then this tolerance essentially becomes a step function.
    margin_value (float): What the value should be when `x` is at 1. Ignored
       if `margin` == 0.
  
  Returns:
    1 if `x` is within bounds. If margin == 0, then this function will return 
    0 for all values of `x` that fall out of bounds. Otherwise, the slope
    of the reward function decline is controlled by `margin` and `margin_value`.

    This creates a gaussian-sloped decline, so the `margin_value` is the value
    you want outputted when `x==1`. You can control the scaling of this slope
    with `margin`.
  """
  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound ({}) is greater than upper bound ({})'.format(
                     lower, upper))
  if margin < 0:
    raise ValueError('Margin must be non-negative: {}'.format(margin))

  if not 0 < margin_value <= 1:
    raise ValueError('Margin value must be valued in (0, 1]: {}'.format(
      margin_value))

  within_bounds = np.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = np.where(within_bounds, 1., 0.)
  else:
    # Compute a simple gaussian decline. More details can be found at
    # https://en.wikipedia.org/wiki/Normal_distribution#Probability_density_function
    scale = np.sqrt(-2 * np.log(margin_value))
    sigmas = np.where(x < lower, lower - x, x - upper) / margin
    values = np.exp(-0.5 * (sigmas * scale) ** 2)
    
    value = np.where(within_bounds, 1., values)
  
  return float(value) if np.isscalar(x) else value