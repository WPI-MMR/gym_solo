import argparse
import gym

import gym_solo
from gym_solo.envs import solo8v2vanilla


if __name__ == '__main__':
  config = solo8v2vanilla.Solo8VanillaConfig()
  env = gym.make('solo8vanilla-v0', use_gui=True, realtime=True, config=config)