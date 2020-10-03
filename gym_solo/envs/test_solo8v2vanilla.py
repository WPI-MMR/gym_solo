import unittest
from gym_solo.envs.solo8v2vanilla_env import Solo8VanillaEnv

import importlib
import os


class TestSolo8v2VanillaEnv(unittest.TestCase):
  def setUp(self):
    self.env = Solo8VanillaEnv()
    
  def testSeed(self):
    seed = 69
    self.env.seed(seed)

    import numpy as np
    import random
    
    numpy_control = float(np.random.rand(1))
    random_control = random.random()

    with self.subTest('random seed'):
      importlib.reload(np)
      importlib.reload(random)

      self.assertNotEqual(numpy_control, float(np.random.rand(1)))
      self.assertNotEqual(random_control, random.random())

    with self.subTest('same seed'):
      importlib.reload(np)
      importlib.reload(random)

      self.env.seed(seed)

      self.assertEqual(numpy_control, float(np.random.rand(1)))
      self.assertEqual(random_control, random.random())
      

if __name__ == '__main__':
  unittest.main()