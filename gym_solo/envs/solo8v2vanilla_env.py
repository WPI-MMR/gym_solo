import pybullet
import gym


class FooEnv(gym.Env):
  metadata = {'render.modes': ['lying']}

  def __init__(self):
    pass

  def step(self, action):
    pass

  def reset(self):
    pass

  def render(self, mode='lying'):
    pass

  def close(self):
    pass