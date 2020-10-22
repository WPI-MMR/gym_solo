import argparse
import gym
import numpy as np

import gym_solo
from gym_solo.envs import solo8v2vanilla


if __name__ == '__main__':
  config = solo8v2vanilla.Solo8VanillaConfig()
  env = gym.make('solo8vanilla-v0', use_gui=True, realtime=True, config=config)

  try:
    print("""\n
          =========================
              Solo 8 v2 Vanilla
              
          Simulation Active.
          
          Exit with ^C.
          =========================
          """)

    while True:
      env.step(np.zeros(env.action_space.shape))
  except KeyboardInterrupt:
    pass