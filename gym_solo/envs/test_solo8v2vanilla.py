import unittest
from gym_solo.envs import solo8v2vanilla as solo_env

from gym import error, spaces
from parameterized import parameterized
from unittest import mock

import importlib
import os
import pybullet as p


class TestSolo8v2VanillaEnv(unittest.TestCase):
  def setUp(self):
    self.env = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig())
    
  def test_seed(self):
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

  @parameterized.expand([
    ('default', {}, p.DIRECT),
    ('nogui', {'use_gui': False}, p.DIRECT),
    ('gui', {'use_gui': True}, p.GUI),
  ])
  @mock.patch('pybullet.connect')
  def test_GUI(self, name, kwargs, expected_ui, mock_connect):
    env = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig(),
                                   **kwargs)
    mock_connect.assert_called_with(expected_ui)

  def test_action_space(self):
    limit = 0.5
    joint_cnt = 12  # 8 dof + 4 "ankle" joints

    space = spaces.Box(-limit, limit, shape=(joint_cnt,))

    config = solo_env.Solo8VanillaConfig(
      motor_torque_limit = limit)
    env = solo_env.Solo8VanillaEnv(config=config)
    
    self.assertEqual(env.action_space, space)

  def test_no_action(self):
    pass


if __name__ == '__main__':
  unittest.main()