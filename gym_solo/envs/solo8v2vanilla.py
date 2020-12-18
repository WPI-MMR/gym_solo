from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pkg_resources
import pybullet as p
import pybullet_data as pbd
import pybullet_utils.bullet_client as bc
import random
import time

import gym
from gym import error, spaces

from gym_solo.core.configs import Solo8BaseConfig
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo.core import termination as terms

from gym_solo import solo_types


@dataclass
class Solo8VanillaConfig(Solo8BaseConfig):
  urdf_path: str = 'assets/solo8v2/solo.urdf'
  starting_joint_pos = {
    'FL_HFE': np.pi / 2,
    'FL_KFE': np.pi,
    'FL_ANKLE': 0,
    'FR_HFE': np.pi / 2,
    'FR_KFE': np.pi,
    'FR_ANKLE': 0,
    'HL_HFE': -np.pi / 2,
    'HL_KFE': -np.pi,
    'HL_ANKLE': 0,
    'HR_HFE': -np.pi / 2,
    'HR_KFE': -np.pi,
    'HR_ANKLE': 0
  }


class Solo8VanillaEnv(gym.Env):
  """An unmodified solo8 gym environment.
  
  Note that the model corresponds to the solo8v2.
  """
  def __init__(self, use_gui: bool = False, realtime: bool = False, 
               config=None, **kwargs) -> None:
    """Create a solo8 env"""
    self._realtime = realtime
    self._config = config

    self.client = bc.BulletClient(
      connection_mode=p.GUI if use_gui else p.DIRECT)
    self.client.setAdditionalSearchPath(pbd.getDataPath())
    self.client.setGravity(*self._config.gravity)
    self.client.setPhysicsEngineParameter(fixedTimeStep=self._config.dt, 
                                          numSubSteps=1)

    self.plane = self.client.loadURDF('plane.urdf')
    self.robot, joint_cnt = self._load_robot()

    self.obs_factory = obs.ObservationFactory(self.client)
    self.reward_factory = rewards.RewardFactory()
    self.termination_factory = terms.TerminationFactory()

    self._zero_gains = np.zeros(joint_cnt)
    self.action_space = spaces.Box(-self._config.max_motor_rotation, 
                                   self._config.max_motor_rotation,
                                   shape=(joint_cnt,))
    
    self.reset(init_call=True)

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
    self.client.setJointMotorControlArray(
      self.robot, np.arange(self.action_space.shape[0]), p.POSITION_CONTROL, 
      targetPositions = action, 
      forces = [self._config.motor_torque_limit] * self.action_space.shape[0])
    self.client.stepSimulation()

    if self._realtime:
      time.sleep(self._config.dt)

    obs_values, obs_labels = self.obs_factory.get_obs()
    reward = self.reward_factory.get_reward()

    # TODO: Write tests for this call
    done = self.termination_factory.is_terminated()

    return obs_values, reward, done, {'labels': obs_labels}

  def reset(self, init_call: bool = False) -> solo_types.obs:
    """Reset the state of the environment and returns an initial observation.
    
    Returns:
      solo_types.obs: The initial observation of the space.
    """
    self.client.removeBody(self.robot)
    self.robot, joint_cnt = self._load_robot()
    
    joint_ordering = [self.client.getJointInfo(self.robot, j)[1].decode('UTF-8')
                      for j in range(joint_cnt)]
    positions = [self._config.starting_joint_pos[j] for j in joint_ordering]

    # Let the robot lay down flat, as intended. Note that this is a hack
    # around modifying the URDF, but that should really be handled in the
    # solidworks phase
    for i in range(500):
      self.client.setJointMotorControlArray(
        self.robot, np.arange(self.action_space.shape[0]), p.POSITION_CONTROL, 
        targetPositions = positions, 
        forces = [self._config.motor_torque_limit] * self.action_space.shape[0])

      self.client.stepSimulation()
    
    if init_call:
      return np.empty(shape=(0,)), []
    else:
      obs_values, _ = self.obs_factory.get_obs()
      return obs_values
  
  @property
  def observation_space(self):
    # TODO: Write tests for this function
    return self.obs_factory.get_observation_space()

  def _load_robot(self) -> Tuple[int, int]:
    """Load the robot from URDF and reset the dynamics.

    Returns:
        Tuple[int, int]: the id of the robot object and the number of joints.
    """
    robot_id = self.client.loadURDF(
      self._config.urdf, self._config.robot_start_pos, 
      self.client.getQuaternionFromEuler(
        self._config.robot_start_orientation_euler),
      flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=False)

    joint_cnt = self.client.getNumJoints(robot_id)
    for joint in range(joint_cnt):
      self.client.changeDynamics(robot_id, joint, 
                                 linearDamping=self._config.linear_damping, 
                                 angularDamping=self._config.angular_damping, 
                                 restitution=self._config.restitution, 
                                 lateralFriction=self._config.lateral_friction)

    return robot_id, joint_cnt

  def _close(self) -> None:
    """Soft shutdown the environment. """
    self.client.disconnect()

  def _seed(self, seed: int) -> None:
    """Set the seeds for random and numpy

    Args:
      seed (int): The seed to set
    """
    np.random.seed(seed)
    random.seed(seed)