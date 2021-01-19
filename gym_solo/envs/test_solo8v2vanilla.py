import unittest
from gym_solo.envs import solo8v2vanilla as solo_env

from gym_solo.core import obs as solo_obs
from gym_solo.testing import CompliantObs
from gym_solo.testing import SimpleReward
from gym_solo.testing import DummyTermination

from gym import error, spaces
from parameterized import parameterized
from unittest import mock

import importlib
import numpy as np
import os
import pybullet as p
import pybullet_utils.bullet_client as bc


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
    
    env.obs_factory.register_observation(CompliantObs(None))
    env.termination_factory.register_termination(DummyTermination(0, True))
    
    env.step(env.action_space.sample())
    self.assertTrue(mock_time.called)

  @mock.patch('pybullet_utils.bullet_client.BulletClient')
  @mock.patch.object(solo_env.Solo8VanillaEnv, 'reset')
  def test_GUI_default(self, fake_reset, mock_client):
    solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig())
    mock_client.assert_called_with(connection_mode=p.DIRECT)

  def test_action_space(self):
    limit = 2 * np.pi
    joint_cnt = 12  # 8 dof + 4 "ankle" joints

    space = spaces.Box(-limit, limit, shape=(joint_cnt,))

    config = solo_env.Solo8VanillaConfig(
      motor_torque_limit = limit)
    env = solo_env.Solo8VanillaEnv(config=config)
    
    self.assertEqual(env.action_space, space)

  def test_invalid_action_space(self):
    self.env._action_space = None
    with self.assertRaises(ValueError):
      self.env.action_space

  def test_actions(self):
    no_op = np.zeros(self.env.action_space.shape[0])
    
    self.env.obs_factory.register_observation(CompliantObs(None))
    self.env.termination_factory.register_termination(DummyTermination(0, True))
    
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
    self.env.obs_factory.register_observation(CompliantObs(None))
    self.env.termination_factory.register_termination(DummyTermination(0, True))
    
    base_pos, base_or = p.getBasePositionAndOrientation(self.env.robot)
    
    action = np.array([5.] * self.env.action_space.shape[0])
    for _ in range(100):
      self.env.step(action)
    self.assertEqual(
      self.env.termination_factory._terminations[0].reset_counter, 1)
      
    new_pos, new_or = p.getBasePositionAndOrientation(self.env.robot)
    self.assert_array_not_almost_equal(base_pos, new_pos)
    self.assert_array_not_almost_equal(base_or, new_or)

    self.env.reset()
    self.assertEqual(
      self.env.termination_factory._terminations[0].reset_counter, 2)

    new_pos, new_or = p.getBasePositionAndOrientation(self.env.robot)
    np.testing.assert_array_almost_equal(base_pos, new_pos)
    np.testing.assert_array_almost_equal(base_or, new_or)

  def test_step_no_rewards(self):
    env = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig())
    with self.assertRaises(ValueError):
      env.step(np.zeros(self.env.action_space.shape[0]))

  def test_step_simple_reward(self):
    self.env.obs_factory.register_observation(CompliantObs(None))
    self.env.termination_factory.register_termination(DummyTermination(0, True))
    
    obs, reward, done, info = self.env.step(self.env.action_space.sample())
    self.assertEqual(reward, 1)

  def test_disjoint_environments(self):
    env1 = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig())
    env1.obs_factory.register_observation(solo_obs.TorsoIMU(env1.robot))
    env1.obs_factory.register_observation(solo_obs.MotorEncoder(env1.robot))
    env1.reward_factory.register_reward(1, SimpleReward())
    env1.termination_factory.register_termination(DummyTermination(0, True))
    home_position = env1.reset()
    
    for i in range(1000):
      env1.step(env1.action_space.sample())

    env2 = solo_env.Solo8VanillaEnv(config=solo_env.Solo8VanillaConfig())
    env2.obs_factory.register_observation(solo_obs.TorsoIMU(env2.robot))
    env2.obs_factory.register_observation(solo_obs.MotorEncoder(env2.robot))
    env2.reward_factory.register_reward(1, SimpleReward())
    env2.termination_factory.register_termination(DummyTermination(0, True))

    np.testing.assert_array_almost_equal(home_position, env2.reset())

if __name__ == '__main__':
  unittest.main()