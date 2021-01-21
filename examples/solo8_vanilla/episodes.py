""" A gentle demo which runs the simulation for 1000 timesteps for 10 episodes.
"""
import gym
import numpy as np

import gym_solo
from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo.core import termination as terms


if __name__ == '__main__':
  config = solo8v2vanilla.Solo8VanillaConfig()
  env = gym.make('solo8vanilla-v0', use_gui=True, realtime=True, config=config)

  env.obs_factory.register_observation(obs.TorsoIMU(env.robot))
  env.reward_factory.register_reward(1,rewards.UprightReward(env.robot))
  env.termination_factory.register_termination(terms.TimeBasedTermination(1000))

  try:
    print("""\n
          =============================================
              Solo 8 v2 Vanilla Episode Test
              
          Simulation Active.
          
          Exit with ^C.
          =============================================
          """)

    for i in range(10):
      env.reset()
      done = False
      while not done:
        _, _, done, _ = env.step(np.zeros(env.action_space.shape))
      print('Episode {} done'.format(i))

  except KeyboardInterrupt:
    pass