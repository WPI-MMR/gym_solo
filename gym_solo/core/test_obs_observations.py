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
                                  30 * 180 / np.pi, 
                                    60 * 180 / np.pi, 90* 180 / np.pi])

    with self.subTest('radians'):
      o = obs.TorsoIMU(0, degrees=False)
      np.testing.assert_allclose(o.compute(),
                                 [0, 0, 1/2 * np.pi,
                                  30, 60, 90,
                                  30, 60, 90])


class TestMotorEncoder(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestMotorEncoder, self).__init__(*args, **kwargs)
    
    # This output is obtained from getJointInfo (pybullet) for all the joints. 
    # This is only for the solo8vanilla. The output from pybullet is converted
    # into a list for mockings
    self.joint_info = [(0, b'FL_HFE', 0, 7, 6, 1, 0.0, 0.0, -10.0, 10.0,
     1000.0, 1000.0, b'FL_UPPER_LEG', (0.0, 0.9928065522152947, -0.11972948625288478), 
     (0.19, 0.1046, 0.0), (-0.05997269288895365, 0.0, 0.0, 0.998200018086379), -1), 
    (1, b'FL_KFE', 0, 8, 7, 1, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'FL_LOWER_LEG', (0.0, 0.9996672990385953, -0.025793240061682075), 
      (-1.377e-05, 0.00822816082925067, -0.08287430545789741), 
      (0.04709322720802768, 0.0, 0.0, 0.9988904984787538), 0), 
    (2, b'FL_ANKLE', 4, -1, -1, 0, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'FL_FOOT', (0.0, 0.0, 0.0), (0.0, -0.0017005235902268143, 
      -0.07069750911605854), (0.012897692844160934, 0.0, 0.0, 0.9999168213003008), 1), 
    (3, b'FR_HFE', 0, 9, 8, 1, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'FR_UPPER_LEG', (0.0, 0.9928065522152947, 0.11972948625288478), 
        (0.19, -0.1046, 0.0), (0.05997269288895365, 0.0, 0.0, 0.998200018086379), -1), 
    (4, b'FR_KFE', 0, 10, 9, 1, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'FR_LOWER_LEG', (0.0, 0.9996672990385953, 0.025793240061682075), 
      (1.377e-05, -0.00822816082925067, -0.08287430545789741), 
      (-0.04709322720802768, 0.0, 0.0, 0.9988904984787538), 3), 
    (5, b'FR_ANKLE', 4, -1, -1, 0, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'FR_FOOT', (0.0, 0.0, 0.0), (0.0, -0.014047115411452295, 
      -0.07110382693156142), (-0.012897692844160934, 0.0, 0.0, 0.9999168213003008), 4), 
    (6, b'HL_HFE', 0, 11, 10, 1, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'HL_UPPER_LEG', (0.0, 0.9928065522152947, -0.11972948625288478), 
      (-0.19, 0.1046, 0.0), (-0.05997269288895365, 0.0, 0.0, 0.998200018086379), -1), 
    (7, b'HL_KFE', 0, 12, 11, 1, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'HL_LOWER_LEG', (0.0, 0.9996672990385953, -0.025793240061682075), 
      (-1.377e-05, 0.00822816082925067, -0.08287430545789741), 
      (0.04709322720802768, 0.0, 0.0, 0.9988904984787538), 6), 
    (8, b'HL_ANKLE', 4, -1, -1, 0, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'HL_FOOT', (0.0, 0.0, 0.0), (0.0, -0.0017005235902268143, 
      -0.07069750911605854), (0.012897692844160934, 0.0, 0.0, 0.9999168213003008), 7), 
    (9, b'HR_HFE', 0, 13, 12, 1, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'HR_UPPER_LEG', (0.0, 0.9928065522152947, 0.11972948625288478), 
      (-0.19, -0.1046, 0.0), (0.05997269288895365, 0.0, 0.0, 0.998200018086379), -1), 
    (10, b'HR_KFE', 0, 14, 13, 1, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'HR_LOWER_LEG', (0.0, 0.9996672990385953, 0.025793240061682075), 
      (1.377e-05, -0.00822816082925067, -0.08287430545789741), 
      (-0.04709322720802768, 0.0, 0.0, 0.9988904984787538), 9), 
    (11, b'HR_ANKLE', 4, -1, -1, 0, 0.0, 0.0, -10.0, 10.0, 1000.0, 
      1000.0, b'HR_FOOT', (0.0, 0.0, 0.0), (0.0, -0.014047115411452295, 
      -0.07110382693156142), (-0.012897692844160934, 0.0, 0.0, 0.9999168213003008), 10)]


  @parameterized.expand([
    ("default", False),
    ("degrees", True)
  ])
  @mock.patch('pybullet.getNumJoints', autospec=True)
  def test_attributes(self, name, degrees, mock_num_joints):
    num_joints = 12
    dummy_robot_id = 0
    mock_num_joints.return_value = num_joints

    o = obs.MotorEncoder(dummy_robot_id, degrees=degrees)
    
    self.assertEqual(o.robot, dummy_robot_id)
    self.assertEqual(o._degrees, degrees)
    self.assertEqual(o._num_joints, num_joints)

  @parameterized.expand([
    ("default", False),
    ("degrees", True)
  ])
  @mock.patch('pybullet.getJointInfo', autospec=True)
  @mock.patch('pybullet.getNumJoints', autospec=True)
  def test_observation_space(self, name, degrees, mock_num_joints, mock_joint_info):

    num_joints = 12
    mock_num_joints.return_value = num_joints
    mock_joint_info.side_effect = lambda robot, joint: self.joint_info[joint]
    dummy_robot_id = 0

    o = obs.MotorEncoder(dummy_robot_id, degrees=degrees)

    position_max = 10

    if degrees:
      position_max = np.degrees(position_max)
    
    position_min = -position_max  

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
    mock_joint_info.side_effect = self.joint_info
    
    ground_truth = ['FL_HFE', 'FL_KFE', 'FL_ANKLE', 'FR_HFE',
     'FR_KFE', 'FR_ANKLE', 'HL_HFE', 'HL_KFE', 'HL_ANKLE',
     'HR_HFE', 'HR_KFE', 'HR_ANKLE']

    o = obs.MotorEncoder(dummy_robot_id)
    self.assertEqual(o.labels(), ground_truth)
    
  @parameterized.expand([
    ("default", False),
    ("degrees", True)
  ])
  @mock.patch('pybullet.getJointState', autospec=True)
  @mock.patch('pybullet.getNumJoints', autospec=True)
  def test_compute(self, name, degrees, mock_num_joints, mock_joint_state):
    dummy_robot_id = 0

    mock_num_joints.return_value = 12

    # This is real case extracted from pybullet
    mock_joint_state.side_effect = [
      (1.5301299626083, 7.554435979113249e-11, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (-3.0853209964046426, -1.4807635358886225e-11, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (1.530127327627307, 7.721743189663525e-12, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (-3.085315909474513, 1.2383787804023483e-11, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (-1.530132288799807, -1.9996261740838904e-11, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (3.0853224548246283, -2.1458507061027347e-11, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (1.5301292310246128, -2.8650758804247376e-11, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (-3.0853176193095613, 1.4177090742904175e-11, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0), 
      (0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0)]
    
    # Built from the mock_joint_state.side_effect value. Refer to getJointState return from
    # https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.p3s2oveabizm
    ground_truth = [1.5301299626083, -3.0853209964046426, 0.0,
      1.530127327627307, -3.085315909474513, 0.0,
      -1.530132288799807, 3.0853224548246283, 0.0,
      1.5301292310246128, -3.0853176193095613, 0.0]

    if degrees:
      ground_truth = np.degrees(ground_truth)

    o = obs.MotorEncoder(dummy_robot_id, degrees=degrees)
    np.testing.assert_allclose(o.compute(), ground_truth)


if __name__ == '__main__':
  unittest.main()