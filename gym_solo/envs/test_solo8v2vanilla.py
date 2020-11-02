import unittest
from gym_solo.envs import solo8v2vanilla as solo_env

from gym_solo.core.test_obs_factory import CompliantObs
from gym_solo.testing import SimpleReward

from gym import error, spaces
from parameterized import parameterized
from unittest import mock

import importlib
import numpy as np
import os
import pybullet as p


class TestSolo8v2VanillaEnv(unittest.TestCase):
  def setUp(self):
    self.env = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig())
    self.env.reward_factory.register_reward(1, SimpleReward())

  def tearDown(self):
    self.env._close()

  def assert_array_not_almost_equal(self, a, b):
    a = np.array(a)
    b = np.array(b)
    
    with self.assertRaises(AssertionError):
      np.testing.assert_array_almost_equal(a, b)

  @mock.patch('time.sleep', autospec=True, return_value=None)
  def test_realtime(self, mock_time):
    env = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig(),
                                   realtime=True)
    env.reward_factory.register_reward(1, SimpleReward())

    env.step(env.action_space.sample())
    self.assertTrue(mock_time.called)
    
  def test_seed(self):
    seed = 69
    self.env._seed(seed)

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

      self.env._seed(seed)

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

  def test_actions(self):
    no_op = np.zeros(self.env.action_space.shape[0])

    # Let the robot stabilize first
    for i in range(1000):
      self.env.step(no_op)

    position, orientation = p.getBasePositionAndOrientation(self.env.robot)

    with self.subTest('no action'):
      for i in range(10):
        self.env.step(no_op)

      new_pos, new_or = p.getBasePositionAndOrientation(self.env.robot)
      np.testing.assert_array_almost_equal(position, new_pos)
      np.testing.assert_array_almost_equal(orientation, new_or)

    with self.subTest('with action'):
      action = np.array([5.] * self.env.action_space.shape[0])
      for i in range(10):
        self.env.step(action)

      new_pos, new_or = p.getBasePositionAndOrientation(self.env.robot)
      self.assert_array_not_almost_equal(position, new_pos)
      self.assert_array_not_almost_equal(orientation, new_or)

  def test_reset(self):
    base_pos, base_or = p.getBasePositionAndOrientation(self.env.robot)
    
    action = np.array([5.] * self.env.action_space.shape[0])
    for i in range(100):
      self.env.step(action)
      
    new_pos, new_or = p.getBasePositionAndOrientation(self.env.robot)
    self.assert_array_not_almost_equal(base_pos, new_pos)
    self.assert_array_not_almost_equal(base_or, new_or)

    self.env.reset()

    new_pos, new_or = p.getBasePositionAndOrientation(self.env.robot)
    np.testing.assert_array_almost_equal(base_pos, new_pos)
    np.testing.assert_array_almost_equal(base_or, new_or)

  def test_step_no_rewards(self):
    env = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig())
    with self.assertRaises(ValueError):
      env.step(np.zeros(self.env.action_space.shape[0]))

  def test_step_simple_reward(self):
    obs, reward, done, info = self.env.step(self.env.action_space.sample())
    self.assertEqual(reward, 1)

  def test_observation_space(self):
    o = CompliantObs(None)
    self.env.obs_factory.register_observation(o)
    self.assertEqual(o.observation_space, self.env.observation_space)

if __name__ == '__main__':
  unittest.main()