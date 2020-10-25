import unittest
from gym_solo.core import obs

from parameterized import parameterized

import numpy as np
import math


class TestObservations(unittest.TestCase):
  @parameterized.expand([
    ('default', 0, False),
    ('degrees', 1, True)
  ])
  def test_torso_imu_attributes(self, name, robot_id, degrees):
    o = obs.TorsoIMU(robot_id, degrees=degrees)
    self.assertEqual(o.robot, robot_id)
    self.assertEqual(o._degrees, degrees)

  @parameterized.expand([
    ('degrees', -180., 180.),
    ('radians', -np.pi, np.pi)
  ])
  def test_torso_imu_bounds(self, name, angle_min, angle_max):
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

    # It seems as if is_bounded requires all dimensions to be bounded:
    # https://github.com/openai/gym/blob/master/gym/spaces/box.py#L66-L67
    self.assertFalse(o.observation_space.is_bounded())


if __name__ == '__main__':
  unittest.main()