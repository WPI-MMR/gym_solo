import unittest
from gym_solo.envs import solo8v2vanilla as solo_env

import importlib
import os


class TestSolo8v2VanillaEnv(unittest.TestCase):
  def setUp(self):
    self.env = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig())
    
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