from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pybullet as p
import time

from gym_solo.core.termination import PerpetualTermination
from gym_solo import solo_types
from gym_solo import testing 

from gym_solo.envs.solo8v2vanilla import Solo8VanillaConfig
from gym_solo.envs.solo8v2vanilla import Solo8VanillaEnv


@dataclass
class RealtimeSolo8VanillaConfig(Solo8VanillaConfig):
  dt: float = None


class RealtimeSolo8VanillaEnv(Solo8VanillaEnv):
  """A realtime solo8 vanilla gym environment.
  
  Note that the model corresponds to the solo8v2. This environment differs from
  a typical Gym environment as the concept of a "step" is relevant in this
  case. In this realtime implementation, the simulation is always running and
  the `step()` function is the same as sending position controls to the 
  robot.
  """
  def __init__(self, use_gui: bool = True, config=None, **kwargs):
    """Create a solo8 env"""
    config = config or RealtimeSolo8VanillaConfig()

    if config.dt is not None:
      raise ValueError('Cannot have a dt in a realtime simulation')

    super().__init__(use_gui, False, config)
    self.reward_factory.register_reward(1, testing.SimpleReward())
    self.termination_factory.register_termination(PerpetualTermination())

  def client_configuration(self):
    """Override the client's `stepSimulation` so all of all of the parent
    functionality can be applied to this sim.
    """
    # Make the step simulation a no-op
    self.client.stepSimulation = lambda *args, **kwargs: solo_types.no_op

  def step(self, action: List[float]):
    """Set the position of the robot's PID controllers to those in "action".

    Args:
      action (List[float]): Motor positions. Note that they are set to
        be in radians.
    """
    self.client.setJointMotorControlArray(
      self.robot, np.arange(self.action_space.shape[0]), p.POSITION_CONTROL, 
      targetPositions = action, 
      forces = [self.config.motor_torque_limit] * self.action_space.shape[0])

  def reset(self, *args, **kwargs):
    """Reset the environment.
    
    In this case, the initial environment is the solo8 with legs folded up
    on the ground.
    """
    super().reset(*args, **kwargs)
    # Since stepSimulation() is a noop, need to use time to reset the bot
    time.sleep(1)

  def get_obs(self):
    """Get the current observation of the environment.

    Note this is different from the OpenAI gym `get_obs()` as this is
    a realtime simulation, so `get_obs()` can have different results even
    if `env.step()` isn't called.
    """
    return self.obs_factory.get_obs()