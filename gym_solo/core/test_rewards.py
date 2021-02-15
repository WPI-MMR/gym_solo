import unittest
from gym_solo.core import rewards
from gym_solo.testing import ReflectiveReward

from parameterized import parameterized
from pybullet_utils import bullet_client
from unittest import mock

import numpy as np
import pybullet as p
import math


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


class TestAdditiveReward(unittest.TestCase):
  def test_empty(self):
    r = rewards.AdditiveReward()
    self.assertListEqual(r._terms, [])

    with self.assertRaises(ValueError):
      r.compute()

  def test_client_passthrough(self):
    client = "client"
    r = rewards.AdditiveReward()
    r.client = client

    sub_r0 = ReflectiveReward(1)
    sub_r1 = ReflectiveReward(1)

    r.add_term(1, sub_r0)
    r.add_term(1, sub_r1)

    self.assertEqual(sub_r0.client, client)
    self.assertEqual(sub_r1.client, client)

  @parameterized.expand([
    ('single_simple', [(1, 1)], 1),
    ('multiple_simple', [(1, 1), (1, 1)], 2),
    ('multiple_float_coeff', [(.5, 1), (.5, 1)], 1),
    ('multiple_mixed_coeff', [(1, 1), (.5, 1)], 1.5),
    ('multiple_with_0', [(1, 1), (.5, 1), (0, 50)], 1.5),
  ])
  def test_compute(self, name, terms, expected_sum):
    r = rewards.AdditiveReward()
    r.client = 'fake_client'
    for coeff, value in terms:
      r.add_term(coeff, ReflectiveReward(value))
    self.assertEqual(expected_sum, r.compute())

  def test_empty_terms(self):
    r = rewards.AdditiveReward()
    with self.assertRaises(ValueError):
      r.compute()


class TestSmallControlReward(unittest.TestCase):
  def test_init(self):
    margin = .25
    robot_id = 5

    r = rewards.SmallControlReward(robot_id, margin=margin)

    self.assertEqual(r._margin, margin)
    self.assertEqual(r._robot_id, robot_id)

  @parameterized.expand([
    ('not_moving_no_margin', 0, 0, 1, 1),
    ('little_movement_no_margin', 0, 1e-5, 0, 0),
    ('little_movement_margin', 1, 1e-5, 0.9, 1),
  ])
  def test_computation(self, name, margin, average_jnt_velocity, min_val, 
                       max_val):
    robot_id = 69
    r = rewards.SmallControlReward(robot_id, margin=margin)
    r.client = mock.MagicMock()
    r.client.getJointState.return_value = (None, average_jnt_velocity)

    reward = r.compute()
    self.assertGreaterEqual(reward, min_val),
    self.assertLessEqual(reward, max_val)

    
class TestHorizontalMoveSpeedReward(unittest.TestCase):
  def test_init(self):
    robot_id = 5
    target_speed = 0
    hard_margin = 20
    soft_margin = 12

    r = rewards.HorizontalMoveSpeedReward(robot_id, target_speed, hard_margin,
                                          soft_margin)
    
    self.assertEqual(robot_id, r._robot_id)
    self.assertEqual(target_speed, r._target_speed)
    self.assertEqual(hard_margin, r._hard_margin)
    self.assertEqual(soft_margin, r._soft_margin)

  @parameterized.expand([
    ('perfectly_still', 0, 0, 0, 0, 1, 1),
    ('within_hard_bounds', .5, 0, 1, 0, 1, 1),
    ('at_hard_bounds', .5, 0, .5, 0, 1, 1),
    ('close_to_hard_bounds_no_soft', .5, 0, .49, 0, 0, 0),
    ('close_to_hard_bounds_soft', .5, 0, .49, 1, .95, 1),
    ('at_soft', .5, 0, 0, .5, 0, 0.05),
  ])
  def test_computation(self, name, speed, target, hard, soft, low, high):
    mock_client = mock.MagicMock()
    mock_client.getBaseVelocity.return_value = (speed / math.sqrt(2), 
                                                speed / math.sqrt(2), None), None
    r = rewards.HorizontalMoveSpeedReward(1, target, hard, soft)
    r.client = mock_client

    val = r.compute()

    self.assertGreaterEqual(val, low)
    self.assertLessEqual(val, high)

    
class TestTorsoHeightReward(unittest.TestCase):
  def test_init(self):
    robot_id = 5
    target_height = .3
    hard_margin = 20
    soft_margin = 12

    r = rewards.TorsoHeightReward(robot_id, target_height, hard_margin, 
                                  soft_margin)
    
    self.assertEqual(robot_id, r._robot_id)
    self.assertEqual(target_height, r._target_height)
    self.assertEqual(hard_margin, r._hard_margin)
    self.assertEqual(soft_margin, r._soft_margin)
    
  @parameterized.expand([
    ('perfect_stand', 1, 1, 0, 0, 1, 1),
    ('within_hard_bounds', .75, 1, .5, 0, 1, 1),
    ('at_hard_bounds', .5, 1, .5, 0, 1, 1),
    ('close_to_hard_bounds_no_soft', .5, 0, .49, 0, 0, 0),
    ('close_to_hard_bounds_soft', .5, 0, .49, 1, .95, 1),
    ('at_soft', .5, 0, 0, .5, 0, 0.05),
  ])
  def test_computation(self, name, height, target, hard, soft, low, high):
    mock_client = mock.MagicMock()
    mock_client.getBasePositionAndOrientation.return_value = (None, None, height), None
    r = rewards.TorsoHeightReward(1, target, hard, soft)
    r.client = mock_client

    val = r.compute()

    self.assertGreaterEqual(val, low)
    self.assertLessEqual(val, high)

    
class TestFlatTorsoReward(unittest.TestCase):
  def test_init(self):
    robot_id = 5
    hard_margin = 20
    soft_margin = 12

    r = rewards.FlatTorsoReward(robot_id, hard_margin, soft_margin)
    
    self.assertEqual(robot_id, r._robot_id)
    self.assertEqual(hard_margin, r._hard_margin)
    self.assertEqual(soft_margin, r._soft_margin)

  @parameterized.expand([
    ('perfect_flat', 0, 0, 0, 1, 1),
    ('within_hard_bounds', .75, 1, 0, 1, 1),
    ('at_hard_bounds', .5, .5, 0, 1, 1),
    ('close_to_hard_bounds_no_soft', .5, .49, 0, 0, 0),
    ('close_to_hard_bounds_soft', .5, .49, 1, .95, 1),
    ('at_soft', .5, 0, .5, 0, 0.05),
  ])
  def test_computation(self, name, theta, hard, soft, low, high):
    mock_client = mock.MagicMock()
    mock_client.getBasePositionAndOrientation.return_value = None, None
    mock_client.getEulerFromQuaternion.return_value = (theta / math.sqrt(2),
                                                       theta / math.sqrt(2),
                                                       None)
    r = rewards.FlatTorsoReward(1, hard, soft)
    r.client = mock_client

    val = r.compute()

    self.assertGreaterEqual(val, low)
    self.assertLessEqual(val, high)


class TestRewardUtilities(unittest.TestCase):
  @parameterized.expand([
    ('simple_in_bounds', 0, (-1, 1), 0, 1e-6, 1),
    ('simple_out_of_bounds', 2, (-1, 1), 0, 1e-6, 0),
    ('in_single_bound', 2, (2, 2), 0, 1e-6, 1),
    ('out_of_single_bound', 2.0001, (2, 2), 0, 1e-6, 0),
    ('at_bounds_edge', 1, (-1, 1), 1, 1e-6, 1),
    ('at_margin_default_value', 2, (-1, 1), 1, 1e-8, 1e-8),
    ('at_margin_margin_value', 2, (-1, 1), 1, .25, .25),
  ])
  def test_tolerance(self, name, x, bounds, margin, margin_value, 
                     expected_value):
    # Floating point issuses cause flakiness when doing an exact comparision
    self.assertAlmostEqual(rewards.tolerance(x, bounds, margin, margin_value),
                     expected_value)

  def test_tolerance_relative(self):
    bounds = (0,0)
    margin = 1.
    margin_value = .25

    val1 = rewards.tolerance(0, bounds, margin, margin_value)
    self.assertEqual(val1, 1)

    val1 = rewards.tolerance(.25, bounds, margin, margin_value)
    val2 = rewards.tolerance(1, bounds, margin, margin_value)

    self.assertAlmostEqual(val2, margin_value)
    self.assertGreater(val1, val2)

  def test_tolerance_bounds_error(self):
    with self.assertRaises(ValueError):
      rewards.tolerance(0, bounds=(1, 0))

  def test_tolerance_margin_error(self):
    with self.assertRaises(ValueError):
      self.assertRaises(rewards.tolerance(0, margin=-1))

  def test_tolerance_margin_value_error(self):
    with self.assertRaises(ValueError):
      self.assertRaises(rewards.tolerance(0, margin_value=0))

if __name__ == '__main__':
  unittest.main()