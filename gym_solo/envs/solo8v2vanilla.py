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
from gym_solo.core import obs
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

    self.obs_factory = obs.ObservationFactory()

    self._client = p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pbd.getDataPath())
    p.setGravity(*self._config.gravity)
    p.setPhysicsEngineParameter(fixedTimeStep=self._config.dt, numSubSteps=1)

    self.plane = p.loadURDF('plane.urdf')
    self.robot, joint_cnt = self._load_robot()

    self._zero_gains = np.zeros(joint_cnt)
    self.action_space = spaces.Box(-self._config.motor_torque_limit, 
                                   self._config.motor_torque_limit,
                                   shape=(joint_cnt,))
    
    self.reset()

  def step(self, action: List[float]) -> Tuple[solo_types.obs, float, bool, 
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
    p.setJointMotorControlArray(self.robot, 
                                np.arange(self.action_space.shape[0]),
                                p.TORQUE_CONTROL, forces=action,
                                positionGains=self._zero_gains, 
                                velocityGains=self._zero_gains)
    p.stepSimulation()

    if self._realtime:
      time.sleep(self._config.dt)

  def reset(self) -> solo_types.obs:
    """Reset the state of the environment and returns an initial observation.
    
    Returns:
      solo_types.obs: The initial observation of the space.
    """
    p.removeBody(self.robot)
    self.robot, _ = self._load_robot()

    # Let gravity do it's thing and reset the environment
    for i in range(1000):
      self.step(self._zero_gains)
    
    # TODO: Return observations for the state
    return []
  
  @property
  def observation_space(self):
    # TODO: Dynamically generate this from the observation factory.
    pass

  def _load_robot(self) -> Tuple[int, int]:
    """Load the robot from URDF and reset the dynamics.

    Returns:
        Tuple[int, int]: the id of the robot object and the number of joints.
    """
    robot_id = p.loadURDF(
      self._config.urdf, self._config.robot_start_pos, 
      p.getQuaternionFromEuler(self._config.robot_start_orientation_euler),
      flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=False)

    joint_cnt = p.getNumJoints(robot_id)
    p.setJointMotorControlArray(robot_id, np.arange(joint_cnt),
                                p.VELOCITY_CONTROL, forces=np.zeros(joint_cnt))

    for joint in range(joint_cnt):
      p.changeDynamics(robot_id, joint, 
                       linearDamping=self._config.linear_damping,
                       angularDamping=self._config.angular_damping,
                       restitution=self._config.restitution,
                       lateralFriction=self._config.lateral_friction)

    return robot_id, joint_cnt

  def _close(self) -> None:
    """Soft shutdown the environment. """
    p.disconnect()

  def _seed(self, seed: int) -> None:
    """Set the seeds for random and numpy

    Args:
      seed (int): The seed to set
    """
    np.random.seed(seed)
    random.seed(seed)