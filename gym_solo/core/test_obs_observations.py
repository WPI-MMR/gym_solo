import unittest
from gym_solo.core import obs

from abc import ABC, abstractmethod
from parameterized import parameterized
from pybullet_utils import bullet_client
from unittest import mock

import numpy as np
import pybullet as p


class ObservationBaseTestCase(unittest.TestCase, ABC):
  @property
  @abstractmethod
  def obs_cls(self):
    """The observation class to instantiate."""
    pass

  def setUp(self):
    self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)

  def tearDown(self):
    self.client.disconnect()

  def build_obs(self, *args, **kwargs):
    o = self.obs_cls(*args, **kwargs)
    o.client = self.client
    return o


class TestTorsoIMU(ObservationBaseTestCase):
  obs_cls = obs.TorsoIMU

  @parameterized.expand([
    ('default', 0, False),
    ('degrees', 1, True)
  ])
  def test_attributes(self, name, robot_id, degrees):
    max_lin = 50
    max_ang = 200

    o = self.build_obs(robot_id, degrees=degrees, max_lin_velocity=max_lin,
                       max_angular_velocity=max_ang)
    self.assertEqual(o.robot, robot_id)
    self.assertEqual(o._degrees, degrees)
    self.assertEqual(o._max_lin, max_lin)
    self.assertEqual(o._max_ang, max_ang)

  @parameterized.expand([
    ('degrees', -180., 180.),
    ('radians', -np.pi, np.pi)
  ])
  def test_bounds(self, name, angle_min, angle_max):
    o = self.build_obs(0, degrees=(name=='degrees'))

    upper = np.array([angle_max] * 3 + [o._max_lin] * 3 + [o._max_ang] * 3)
    lower = np.array([angle_min] * 3 + [-o._max_lin] * 3 + [-o._max_ang] * 3)

    np.testing.assert_allclose(o.observation_space.low, lower)
    np.testing.assert_allclose(o.observation_space.high, upper)

    # It seems as if is_bounded() requires all dimensions to be bounded:
    # https://github.com/openai/gym/blob/master/gym/spaces/box.py#L66-L67
    self.assertTrue(o.observation_space.is_bounded())

  @mock.patch('pybullet.getBasePositionAndOrientation')
  @mock.patch('pybullet.getBaseVelocity')
  def test_compute(self, mock_vel, mock_base):
    mock_base.return_value = (None, 
                              [0, 0, 0.707, 0.707])
    mock_vel.return_value = ([-5, 6, 7],
                              [-.5, .6, .7])

    with self.subTest('degrees'):
      o = self.build_obs(0, degrees=True, max_angular_velocity=180)
      np.testing.assert_allclose(o.compute(),
                                 [0, 0, 90,
                                  -5, 6, 7,
                                  -.5 * 180 / np.pi, 
                                    .6 * 180 / np.pi, .7 * 180 / np.pi])

    with self.subTest('radians'):
      o = self.build_obs(0, degrees=False)
      np.testing.assert_allclose(o.compute(),
                                 [0, 0, 1/2 * np.pi,
                                  -5, 6, 7,
                                  -.5, .6, .7])

  def test_clipping(self):
    mock_client = mock.MagicMock()
    mock_client.getBasePositionAndOrientation.return_value = None, None
    mock_client.getEulerFromQuaternion.return_value = (1, 2, 3)

    max_lin = 2
    max_ang = 3
    o = self.build_obs(0, max_lin_velocity = max_lin,
                       max_angular_velocity = max_ang)
    o.client = mock_client

    with self.subTest('postive values'):
      mock_client.getBaseVelocity.return_value = ((100, 100, 100), 
                                                  (200, 200, 200))
      obs = o.compute()
      np.testing.assert_array_equal(obs[3:], [max_lin] * 3 + [max_ang] * 3)

    with self.subTest('negative values'):
      mock_client.getBaseVelocity.return_value = ((-100, -100, -100), 
                                                  (-200, -200, -200))
      obs = o.compute()
      np.testing.assert_array_equal(obs[3:], [-max_lin] * 3 + [-max_ang] * 3)


class TestMotorEncoder(ObservationBaseTestCase):
  obs_cls = obs.MotorEncoder

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
  @mock.patch('pybullet.getNumJoints')
  def test_attributes(self, name, degrees, mock_num_joints):
    num_joints = 12
    dummy_robot_id = 0
    max_motor_rot = 69
    mock_num_joints.return_value = num_joints

    o = self.build_obs(dummy_robot_id, degrees=degrees, 
                       max_rotation=max_motor_rot)
    
    self.assertEqual(o.robot, dummy_robot_id)
    self.assertEqual(o._degrees, degrees)
    self.assertEqual(o._num_joints, num_joints)
    self.assertEqual(o._max_rot, max_motor_rot)

  @parameterized.expand([
    ("default", False),
    ("degrees", True)
  ])
  @mock.patch('pybullet.getJointInfo')
  @mock.patch('pybullet.getNumJoints')
  def test_observation_space(self, name, degrees, mock_num_joints, mock_joint_info):

    num_joints = 12
    mock_num_joints.return_value = num_joints
    mock_joint_info.side_effect = lambda robot, joint: self.joint_info[joint]
    dummy_robot_id = 0

    o = self.build_obs(dummy_robot_id, degrees=degrees)

    position_max = 10

    if degrees:
      position_max = np.degrees(position_max)
    
    position_min = -position_max  

    lower_bound = np.full(num_joints, position_min)
    upper_bound = np.full(num_joints, position_max)   

    np.testing.assert_allclose(o.observation_space.low, lower_bound)
    np.testing.assert_allclose(o.observation_space.high, upper_bound)

  def test_artifical_obs_space(self):
    joints = 12
    max_rot = 6.9
    mock_client = mock.MagicMock()
    mock_client.getNumJoints.return_value = joints

    o = self.build_obs(0, max_rotation=max_rot)
    o.client = mock_client

    obs_space = o.observation_space

    np.testing.assert_array_equal(
      obs_space.low, np.array([-max_rot] * joints, dtype=np.float32))
    np.testing.assert_array_equal(
      obs_space.high, np.array([max_rot] * joints, dtype=np.float32))

  @mock.patch('pybullet.getJointInfo')
  @mock.patch('pybullet.getNumJoints')
  def test_labels(self, mock_num_joints, mock_joint_info):
    num_joints = 12
    dummy_robot_id = 0

    mock_num_joints.return_value = num_joints
    mock_joint_info.side_effect = self.joint_info
    
    ground_truth = ['FL_HFE', 'FL_KFE', 'FL_ANKLE', 'FR_HFE',
     'FR_KFE', 'FR_ANKLE', 'HL_HFE', 'HL_KFE', 'HL_ANKLE',
     'HR_HFE', 'HR_KFE', 'HR_ANKLE']

    o = self.build_obs(dummy_robot_id)
    self.assertEqual(o.labels, ground_truth)
    
  @parameterized.expand([
    ("default", False),
    ("degrees", True)
  ])
  @mock.patch('pybullet.getJointState')
  @mock.patch('pybullet.getNumJoints')
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

    o = self.build_obs(dummy_robot_id, degrees=degrees)
    np.testing.assert_allclose(o.compute(), ground_truth)

  def test_clipping(self):
    joints = 12
    max_rot = .5

    mock_client = mock.MagicMock()
    mock_client.getNumJoints.return_value = joints

    o = self.build_obs(0, max_rotation=max_rot)
    o.client = mock_client

    with self.subTest('positive values'):
      mock_client.getJointState.return_value = 69, None
      np.testing.assert_array_equal(
        o.compute(), np.array([max_rot] * joints, dtype=np.float32))

    with self.subTest('negative values'):
      mock_client.getJointState.return_value = -69, None
      np.testing.assert_array_equal(
        o.compute(), np.array([-max_rot] * joints, dtype=np.float32))


if __name__ == '__main__':
  unittest.main()