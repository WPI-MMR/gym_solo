import unittest
from gym_solo.core import rewards

from parameterized import parameterized


class TestReward(rewards.Reward):
  def __init__(self, return_value):
    self._return_value = return_value

  def compute(self):
    return self._return_value


class TestRewardsFactory(unittest.TestCase):
  def test_empty(self):
    rf = rewards.RewardFactory()
    self.assertListEqual(rf._rewards, [])

  @parameterized.expand([
    ('single', {1: 2.5}, 2.5),
    ('two_happy', {1: 1, 2: 2}, 5),
    ('0-weight', {0: 1, 2: 2}, 4),
    ('three', {1: 1, 2: 2, 3: 3}, 14),
  ])
  def test_register_and_compute(self, name, rewards_dict, expected_reward):
    rf = rewards.RewardFactory()
    for weight, reward in rewards_dict.items():
      rf.register_reward(weight, TestReward(reward))
    self.assertEqual(rf.get_reward(), expected_reward)


if __name__ == '__main__':
  unittest.main()