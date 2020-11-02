import unittest
from gym_solo.core import rewards

from parameterized import parameterized
from unittest import mock

import numpy as np


class TestReward(rewards.Reward):
  def __init__(self, return_value):
    self._return_value = return_value

  def compute(self):
    return self._return_value


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
      rf.register_reward(weight, TestReward(reward))
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


if __name__ == '__main__':
  unittest.main()