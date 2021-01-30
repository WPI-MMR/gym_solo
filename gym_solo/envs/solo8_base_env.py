from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod

import gym
import numpy as np
import pybullet as p
import pybullet_data as pbd
import pybullet_utils.bullet_client as bc
import random

from gym_solo.core import termination as terms
from gym_solo.core import configs
from gym_solo.core import obs
from gym_solo.core import rewards
import gym_solo.solo_types as solo_types


class Solo8BaseEnv(ABC, gym.Env):
  """Solo 8 abstract base environment."""
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self, config: configs.Solo8BaseConfig, use_gui: bool):
    """Create a solo8 env.

    Args:
      config (configs.Solo8BaseConfig): The SoloConfig. Defaults to None.
      use_gui (bool): Whether or not to show the pybullet GUI. Defaults to 
        False.
    """
    self.config = config

    self.client = bc.BulletClient(
      connection_mode=p.GUI if use_gui else p.DIRECT)
    self.client.setAdditionalSearchPath(pbd.getDataPath())
    self.client.setGravity(*self.config.gravity)

    if self.config.dt:
      self.client.setPhysicsEngineParameter(fixedTimeStep=self.config.dt, 
                                            numSubSteps=1)
    else:
      self.client.setRealTimeSimulation(1)

    self.plane = self.client.loadURDF('plane.urdf')
    self.load_bodies()

    self.obs_factory = obs.ObservationFactory(self.client)
    self.reward_factory = rewards.RewardFactory(self.client)
    self.termination_factory = terms.TerminationFactory()

    self.reset(init_call=True)

  @abstractmethod
  def load_bodies(self):
    """Load the bodies into the environment. 
    
    Note that a plane has already been loaded in and the entire environment
    is encapsulated within the self.client object. Thus, all bodies should
    be added via the self.client interface.
    """
    pass

  @property
  @abstractmethod
  def action_space(self) -> gym.Space:
    """Get the action space of the agent.

    Returns:
      gym.Space: A Space representing the domain of valid moves for the
        agent.
    """
    pass

  @abstractmethod
  def reset(self, init_call: bool = False):
    """Reset the environment.
    
    For best results, this method should be deterministic; i.e. the environment
    should return to the same state everytime this method is called.

    Args:
      init_call (bool, optional): If this function is being called from the init
        function. Defaults to False.
    """
    pass

  @abstractmethod
  def step(self, action: List[float]) -> Tuple[solo_types.obs, float, bool, 
                                                Dict[Any, Any]]:
    """Have the agent apply the action in the environment.

    Args:
        action (List[float]): The action for the agent to take. Requires that
        this conforms to self.action_space.

    Returns:
      Tuple[solo_types.obs, float, bool, Dict[Any, Any]]: A tuple of the next
        observation, the reward for that step, whether or not the episode 
        terminates, and an info dict for misc diagnostic details.
    """
    pass

  @property
  def observation_space(self) -> gym.Space:
    """Get the agent's observation space.

    Returns:
      gym.Space: The agent's observation space.
    """
    return self.obs_factory.get_observation_space()

  def render(self, mode='rgb_array') -> List[List[List[int]]]:
    """Render the current state of the Solo 8 env.
    
    Note that the camera is controlled by the `config` option passed in at
    construction time.

    Args:
      mode (str, optional): Rendering mode. Refer to OpenAI Gym documentation. 
        Defaults to 'rgb_array'.

    Returns:
      List[List[List[int]]]: a 2D list of RGBA pixel data.
    """
    proj_matrix = self.client.computeProjectionMatrixFOV(
      self.config.render_fov, self.config.render_aspect,
      0.01, 100)
    view_matrix = self.client.computeViewMatrixFromYawPitchRoll(
      self.config.render_pos, self.config.render_cam_distance, 
      self.config.render_yaw, self.config.render_pitch,
      self.config.render_roll, 2)
    
    w, h, rgb, _, _ = self.client.getCameraImage(
      self.config.render_width, self.config.render_height, view_matrix,
      proj_matrix)

    # 4 Channels for RGBA
    return np.reshape(rgb, (h, w, 4))

  def _close(self):
    """Soft shutdown the environment."""
    self.client.disconnect()

  def _seed(self, seed: int) -> None:
    """Set the seeds for random and numpy

    Args:
      seed (int): The seed to set
    """
    np.random.seed(seed)
    random.seed(seed)