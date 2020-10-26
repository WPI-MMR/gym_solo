""" A demo for the Solo v2 Vanilla with configurable observations

The motors have no torque applied to them; the user can interact with the robot
via the interactive GUI. The simulation will output the observation.
"""
import argparse
import gym
import numpy as np

import gym_solo
from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs


if __name__ == '__main__':
  config = solo8v2vanilla.Solo8VanillaConfig()
  env = gym.make('solo8vanilla-v0', use_gui=True, realtime=True, config=config)

  env.obs_factory.register_observation(obs.TorsoIMU(env.robot))

  try:
    print("""\n
          =============================================
              Solo 8 v2 Vanilla Observation Dump
              
          Simulation Active.
          
          Exit with ^C.
          =============================================
          """)

    while True:
      obs, reward, done, info = env.step(np.zeros(env.action_space.shape))
      print(obs)
  except KeyboardInterrupt:
    pass