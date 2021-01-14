import unittest
from gym_solo.core import rewards
from gym_solo.testing import ReflectiveReward

from parameterized import parameterized
from pybullet_utils import bullet_client
from unittest import mock

import numpy as np
import pybullet as p


class TestRewardsFactory(unittest.TestCase):
  def test_empty(self):
    rf = rewards.RewardFactory(None)
    self.assertListEqual(rf._rewards, [])

    with self.assertRaises(ValueError):
      rf.get_reward()

  def test_unique_clients(self):
    # Using ints as a computation won't actually be run
    client0 = 1
    r0 = ReflectiveReward(0)
    rf0 = rewards.RewardFactory(client0)
    rf0.register_reward(1, r0)

    client1 = 2
    r1 = ReflectiveReward(1)
    rf1 = rewards.RewardFactory(client1)
    rf1.register_reward(1, r1)

    self.assertEqual(client0, r0.client)
    self.assertEqual(client1, r1.client)

  @parameterized.expand([
    ('single', {1: 2.5}, 2.5),
    ('two_happy', {1: 1, 2: 2}, 5),
    ('0-weight', {0: 1, 2: 2}, 4),
    ('negative-weight', {-1: 1, 2: 2}, 3),
    ('three', {1: 1, 2: 2, 3: 3}, 14),
  ])
  def test_register_and_compute(self, name, rewards_dict, expected_reward):
    client = bullet_client.BulletClient(connection_mode=p.DIRECT)
    rf = rewards.RewardFactory(client)
    for weight, reward in rewards_dict.items():
      rf.register_reward(weight, ReflectiveReward(reward))
    self.assertEqual(rf.get_reward(), expected_reward)
    client.disconnect()


class TestRewardInterface(unittest.TestCase):
  def test_no_client(self):
    r = ReflectiveReward(0)
    with self.assertRaises(ValueError):
      r.client


class RewardBaseTestCase(unittest.TestCase):
  def setUp(self):
    self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)

  def tearDown(self):
    self.client.disconnect()

class TestUprightReward(RewardBaseTestCase):
  def test_init(self):
    robot_id = 0
    r = rewards.UprightReward(robot_id)
    r.client = self.client
    self.assertEqual(robot_id, r._robot_id)

  @parameterized.expand([
    ('flat', (0, 0, 0), 0),
    ('upright', (0, 90, 0), -1.),
    ('upside down', (0, -90, 0), 1.),
    ('y dependence', (-45, 90, -90), -1.),
  ])
  @mock.patch('pybullet.getBasePositionAndOrientation')
  @mock.patch('pybullet.getEulerFromQuaternion')
  def test_computation(self, name, orien, expected_reward, mock_euler,
                       mock_orien):
    mock_orien.return_value = None, None

    orien_radians = tuple(i * np.pi / 180 for i in orien)
    mock_euler.return_value = orien_radians

    reward = rewards.UprightReward(None)
    reward.client = self.client

    self.assertEqual(reward.compute(), expected_reward)


class TestHomePositionReward(RewardBaseTestCase):
  def test_init(self):
    robot_id = 0
    r = rewards.HomePositionReward(robot_id)
    r.client = self.client
    self.assertEqual(robot_id, r._robot_id)

  @mock.patch('pybullet.getBasePositionAndOrientation')
  @mock.patch('pybullet.getEulerFromQuaternion')
  def test_computation(self, mock_euler, mock_orien):
    r = rewards.HomePositionReward(0)
    r.client = self.client

    def _mocked_reward(pos, orien):
      mock_orien.return_value = pos, None
      mock_euler.return_value = orien
      return r.compute()

    # Starting position
    start_reward = _mocked_reward((0, 0, 0), (0, 0, 0))

    # Rotated about the z-axis
    z_rot_reward = _mocked_reward((0, 0, 0), (0, 0, .5))

    # Semi standing up at an angle, 
    standing_skewed_reward = _mocked_reward((0, 0, .15), (0, np.pi / 4, 0))

    # Semi standing up completely flat (should be better than skewed)
    standing_straight_reward = _mocked_reward((0, 0, .15), (0, 0, 0))

    # Home position
    home_reward = _mocked_reward(
      (0, 0, rewards.HomePositionReward._quad_standing_height), (0, 0, 0))
    
    self.assertEqual(start_reward, z_rot_reward)
    self.assertLess(start_reward, standing_straight_reward)
    self.assertLess(standing_skewed_reward, standing_straight_reward)
    self.assertLess(standing_straight_reward, home_reward)
    

if __name__ == '__main__':
  unittest.main()