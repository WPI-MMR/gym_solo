import unittest
from gym_solo.core import obs

from parameterized import parameterized
from unittest import mock

import numpy as np
import math


class TestTorsoIMU(unittest.TestCase):
  @parameterized.expand([
    ('default', 0, False),
    ('degrees', 1, True)
  ])
  def test_attributes(self, name, robot_id, degrees):
    o = obs.TorsoIMU(robot_id, degrees=degrees)
    self.assertEqual(o.robot, robot_id)
    self.assertEqual(o._degrees, degrees)

  @parameterized.expand([
    ('degrees', -180., 180.),
    ('radians', -np.pi, np.pi)
  ])
  def test_bounds(self, name, angle_min, angle_max):
    o = obs.TorsoIMU(0, name == 'degrees')

    angles = {'θx', 'θy', 'θz'}
    space_len = len(o.labels)

    upper = np.array([np.inf] * space_len)
    lower = np.array([-np.inf] * space_len)

    for i, lbl in enumerate(o.labels):
      if lbl in angles:
        upper[i] = angle_max
        lower[i] = angle_min

    np.testing.assert_allclose(o.observation_space.low, lower)
    np.testing.assert_allclose(o.observation_space.high, upper)

    # It seems as if is_bounded() requires all dimensions to be bounded:
    # https://github.com/openai/gym/blob/master/gym/spaces/box.py#L66-L67
    self.assertFalse(o.observation_space.is_bounded())

  @mock.patch('pybullet.getBasePositionAndOrientation', autospec=True)
  @mock.patch('pybullet.getBaseVelocity', autospec=True)
  def test_compute(self, mock_vel, mock_base):
    mock_base.return_value = (None, 
                              [0, 0, 0.707, 0.707])
    mock_vel.return_value = ([30, 60, 90],
                             [30, 60, 90])

    with self.subTest('degrees'):
      o = obs.TorsoIMU(0, degrees=True)
      np.testing.assert_allclose(o.compute(),
                                 [0, 0, 90,
                                  30, 60, 90,
                                  30, 60, 90])

    with self.subTest('radians'):
      o = obs.TorsoIMU(0, degrees=False)
      np.testing.assert_allclose(o.compute(),
                                 [0, 0, 1/2 * np.pi,
                                  30, 60, 90,
                                  1/6 * np.pi, 1/3 * np.pi, 1/2 * np.pi])


class TestMotorEncoder(unittest.TestCase):

  @parameterized.expand([
    ("default", 0, False),
    ("degrees", 0, True)
  ])
  @mock.patch('pybullet.getNumJoints', autospec=True)
  def test_attributes(self, name, robot_id, degrees, mock_num_joints):
    num_joints = 11
    mock_num_joints.return_value = num_joints

    o = obs.MotorEncoder(robot_id, degrees= degrees)
    
    self.assertEqual(o.robot, robot_id)
    self.assertEqual(o._degrees, degrees)
    self.assertEqual(o.num_joints, num_joints)

  @parameterized.expand([
    ("default", False),
    ("degrees", True)
  ])
  @mock.patch('pybullet.getNumJoints', autospec=True)
  def test_observation_space(self, name, degrees, mock_num_joints):
    num_joints = 12
    mock_num_joints.return_value = num_joints
    dummy_robot_id = 0

    o = obs.MotorEncoder(dummy_robot_id, degrees= degrees)

    if degrees:
      position_max = 572.96
      position_min = -572.96
    else:
      position_max = 10
      position_min = -10

    lower_bound = np.full(num_joints, position_min)
    upper_bound = np.full(num_joints, position_max)   

    np.testing.assert_allclose(o.observation_space.low, lower_bound)
    np.testing.assert_allclose(o.observation_space.high, upper_bound)


  @mock.patch('pybullet.getJointInfo', autospec=True)
  @mock.patch('pybullet.getNumJoints', autospec=True)
  def test_labels(self, mock_num_joints, mock_joint_info):
    num_joints = 12
    dummy_robot_id = 0

    mock_num_joints.return_value = num_joints
    # This has to change with respect to joint index in the for loop
    mock_joint_info.side_effect = [[None, b'FL_HFE'], 
    [None, b'FL_KFE'], [None, b'FL_ANKLE'], [None, b'FR_HFE'], 
    [None, b'FR_KFE'], [None, b'FR_ANKLE'], [None, b'HL_HFE'], 
    [None, b'HL_KFE'], [None, b'HL_ANKLE'], [None, b'HR_HFE'], 
    [None, b'HR_KFE'], [None, b'HR_ANKLE']]
    
    ground_truth = ['FL_HFE', 'FL_KFE', 'FL_ANKLE', 'FR_HFE',
     'FR_KFE', 'FR_ANKLE', 'HL_HFE', 'HL_KFE', 'HL_ANKLE',
     'HR_HFE', 'HR_KFE', 'HR_ANKLE']

    o = obs.MotorEncoder(dummy_robot_id)
    self.assertEqual(o.labels(), ground_truth)
    

  def test_compute(self):
    pass

if __name__ == '__main__':
  unittest.main()