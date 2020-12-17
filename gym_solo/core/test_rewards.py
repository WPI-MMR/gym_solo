import unittest
from gym_solo.core import rewards
from gym_solo.testing import ReflectiveReward

from parameterized import parameterized
from unittest import mock

import numpy as np


class TestRewardsFactory(unittest.TestCase):
  def test_empty(self):
    rf = rewards.RewardFactory()
    self.assertListEqual(rf._rewards, [])

    with self.assertRaises(ValueError):
      rf.get_reward()

  @parameterized.expand([
    ('single', {1: 2.5}, 2.5),
    ('two_happy', {1: 1, 2: 2}, 5),
    ('0-weight', {0: 1, 2: 2}, 4),
    ('negative-weight', {-1: 1, 2: 2}, 3),
    ('three', {1: 1, 2: 2, 3: 3}, 14),
  ])
  def test_register_and_compute(self, name, rewards_dict, expected_reward):
    rf = rewards.RewardFactory()
    for weight, reward in rewards_dict.items():
      rf.register_reward(weight, ReflectiveReward(reward))
    self.assertEqual(rf.get_reward(), expected_reward)


class TestUprightReward(unittest.TestCase):
  def test_init(self):
    robot_id = 0
    r = rewards.UprightReward(robot_id)
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
    self.assertEqual(reward.compute(), expected_reward)


class TestHomePositionReward(unittest.TestCase):
  def test_init(self):
    robot_id = 0
    r = rewards.HomePositionReward(robot_id)
    self.assertEqual(robot_id, r._robot_id)

  @mock.patch('pybullet.getBasePositionAndOrientation')
  @mock.patch('pybullet.getEulerFromQuaternion')
  def test_computation(self, mock_euler, mock_orien):
    r = rewards.HomePositionReward(0)

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