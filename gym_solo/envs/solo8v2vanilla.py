from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pybullet as p
import pybullet_data as pbd
import pybullet_utils.bullet_client as bc
import random
import time

import gym
from gym import spaces

from gym_solo.core.configs import Solo8BaseConfig
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo.core import termination as terms
from gym_solo.envs import Solo8BaseEnv

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


class Solo8VanillaEnv(Solo8BaseEnv):
  """An unmodified solo8 gym environment.
  
  Note that the model corresponds to the solo8v2.
  """
  def __init__(self, use_gui: bool = False, realtime: bool = False, 
               config=None, **kwargs):
    """Create a solo8 env"""
    self._realtime = realtime
    super().__init__(config or Solo8VanillaConfig(), use_gui)

  @property
  def action_space(self) -> gym.Space:
    """The action space of the agent.
    
    Aka what actions are "legal" for the agent to carry out.

    Raises:
      ValueError: Invalid action space

    Returns:
      gym.Space: The valid actions that the agent can take.
    """
    if self._action_space:
      return self._action_space
    else:
      raise ValueError('No valid action space')

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
      forces = [self.config.motor_torque_limit] * self.action_space.shape[0])
    self.client.stepSimulation()

    if self._realtime:
      time.sleep(self.config.dt)

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
    self.load_bodies()
    
    joint_ordering = [self.client.getJointInfo(self.robot, j)[1].decode('UTF-8')
                      for j in range(self._joint_cnt)]
    positions = [self.config.starting_joint_pos[j] for j in joint_ordering]

    # Let the robot lay down flat, as intended. Note that this is a hack
    # around modifying the URDF, but that should really be handled in the
    # solidworks phase
    for i in range(500):
      self.client.setJointMotorControlArray(
        self.robot, np.arange(self.action_space.shape[0]), p.POSITION_CONTROL, 
        targetPositions = positions, 
        forces = [self.config.motor_torque_limit] * self.action_space.shape[0])

      self.client.stepSimulation()
    
    if init_call:
      return np.empty(shape=(0,)), []
    else:
      obs_values, _ = self.obs_factory.get_obs()
      return obs_values

  def load_bodies(self) -> Tuple[int, int]:
    """Load the robot from URDF and reset the dynamics.

    Returns:
        Tuple[int, int]: the id of the robot object and the number of joints.
    """
    robot_id = self.client.loadURDF(
      self.config.urdf, self.config.robot_start_pos, 
      self.client.getQuaternionFromEuler(
        self.config.robot_start_orientation_euler),
      flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=False)

    joint_cnt = self.client.getNumJoints(robot_id)
    for joint in range(joint_cnt):
      self.client.changeDynamics(robot_id, joint, 
                                 linearDamping=self.config.linear_damping, 
                                 angularDamping=self.config.angular_damping, 
                                 restitution=self.config.restitution, 
                                 lateralFriction=self.config.lateral_friction)
    
    self.robot = robot_id
    self._joint_cnt = joint_cnt
    self._zero_gains = np.zeros(self._joint_cnt)

    self._action_space = spaces.Box(-self.config.max_motor_rotation, 
                                   self.config.max_motor_rotation,
                                   shape=(self._joint_cnt,))
