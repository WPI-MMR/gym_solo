from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pkg_resources
import pybullet as p
import pybullet_data as pbd
import random
import time

import gym
from gym import error, spaces

from gym_solo.core.configs import Solo8BaseConfig
from gym_solo import solo_types


@dataclass
class Solo8VanillaConfig(Solo8BaseConfig):
  urdf_path: str = 'assets/solo8v2/solo.urdf'


class Solo8VanillaEnv(gym.Env):
  """An unmodified solo8 gym environment.
  
  Note that the model corresponds to the solo8v2.
  """
  def __init__(self, use_gui: bool = False, realtime: bool = False, 
               config=None, **kwargs) -> None:
    """Create a solo8 env"""
    self._realtime = realtime
    self._config = config

    self._client = p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pbd.getDataPath())
    p.setGravity(*self._config.gravity)
    p.setPhysicsEngineParameter(fixedTimeStep=self._config.dt, numSubSteps=1)

    self._plane = p.loadURDF('plane.urdf')

    self._robot = p.loadURDF(
      self._config.urdf, self._config.robot_start_pos, 
      p.getQuaternionFromEuler(self._config.robot_start_orientation_euler),
      flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=False)

    joint_cnt = p.getNumJoints(self._robot)
    self._zero_gains = np.zeros(joint_cnt)
    p.setJointMotorControlArray(self._robot, np.arange(joint_cnt),
                                p.VELOCITY_CONTROL, forces=np.zeros(joint_cnt))

    self.action_space = spaces.Box(-self._config.motor_torque_limit, 
                                   self._config.motor_torque_limit,
                                   shape=(joint_cnt,))
    
    for joint in range(joint_cnt):
      p.changeDynamics(self._robot, joint, 
                       linearDamping=self._config.linear_damping,
                       angularDamping=self._config.angular_damping,
                       restitution=self._config.restitution,
                       lateralFriction=self._config.lateral_friction)

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
    p.setJointMotorControlArray(self._robot, 
                                np.arange(self.action_space.shape[0]),
                                p.TORQUE_CONTROL, forces=action,
                                positionGains=self._zero_gains, 
                                velocityGains=self._zero_gains)
    p.stepSimulation()

    if self._realtime:
      time.sleep(self._config.dt)

  def _reset(self) -> solo_types.obs:
    """Reset the state of the environment and returns an initial observation.
    
    Returns:
      solo_types.obs: The initial observation of the space.
    """
    p.resetBasePositionAndOrientation(
      self._robot, self._config.robot_start_pos,
      p.getQuaternionFromEuler(self._config.robot_start_orientation_euler))
    
    # TODO: Return observations for the state
    return []
  
  @property
  def observation_space(self):
    # TODO: Dynamically generate this from the observation factory.
    pass

  def _close(self) -> None:
    """Soft shutdown the environment. """
    p.disconnect()

  def seed(self, seed: int) -> None:
    """Set the seeds for random and numpy

    Args:
      seed (int): The seed to set
    """
    np.random.seed(seed)
    random.seed(seed)