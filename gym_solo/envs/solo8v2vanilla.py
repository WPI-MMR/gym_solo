from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pkg_resources
import pybullet as p
import pybullet_data as pbd
import random

import gym
from gym import error, spaces

from gym_solo.core.configs import Solo8BaseConfig
from gym_solo import solo_types


@dataclass
class Solo8VanillaConfig(Solo8BaseConfig):
  urdf_path: str = 'assets/solo8v2/urdf/solo.urdf'


class Solo8VanillaEnv(gym.Env):
  """An unmodified solo8 gym environment.
  
  Note that the model corresponds to the solo8v2.
  """
  def __init__(self, use_gui: bool = False, realtime: bool = False, 
               config=None, **kwargs) -> None:
    """Create a solo8 env"""
    self.client = p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pbd.getDataPath())
    p.setGravity(*config.gravity)
    p.setPhysicsEngineParameter(fixedTimeStep=config.dt, numSubSteps=1)

    self.plane = p.loadURDF('plane.urdf')

    self.robot = p.loadURDF(
      config.urdf, config.robot_start_pos, 
      p.getQuaternionFromEuler(config.robot_start_orientation_euler),
      flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=False)
    
    joint_cnt = p.getNumJoints(self.robot)
    self.action_space = spaces.Box(-config.motor_torque_limit, 
                                   config.motor_torque_limit,
                                   shape=(joint_cnt,))
    
    for joint in range(joint_cnt):
      p.changeDynamics(self.robot, joint, linearDamping=config.linear_damping,
                       angularDamping=config.angular_damping,
                       restitution=config.restitution,
                       lateralFriction=config.lateral_friction)

  def _step(self, action: List[float]) -> Tuple[solo_types.obs, float, bool, 
                                                Dict[Any, Any]]:
    """The agent takes a step in the environment.

    Args:
      action (List[float]): The torques applied to the motors in N•m. Note
        len(action) == the # of actuator

    Returns:
      Tuple[solo_types.obs, float, bool, Dict[Any, Any]]: A tuple of the next
        observation, the reward for that step, whether or not the episode 
        terminates, and an info dict for misc diagnostic details.
    """
    pass

  def _reset(self) -> solo_types.obs:
    """Reset the state of the environment and returns an initial observation.
    
    Returns:
      solo_types.obs: The initial observation of the space.
    """
    pass
  
  @property
  def observation_space(self):
    pass

  def _close(self) -> None:
    """Soft shutdown the environment. """
    pass

  def seed(self, seed: int) -> None:
    """Set the seeds for random and numpy

    Args:
      seed (int): The seed to set
    """
    np.random.seed(seed)
    random.seed(seed)