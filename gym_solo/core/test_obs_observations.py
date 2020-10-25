import unittest
from gym_solo.core import obs

from parameterized import parameterized


class TestObservations(unittest.TestCase):
  @parameterized.expand([
    ('default', 0, False),
    ('degrees', 1, True)
  ])
  def test_torso_imu_attributes(self, name, robot_id, degrees):
    o = obs.TorsoIMU(robot_id, degrees=degrees)
    self.assertEqual(o.robot, robot_id)
    self.assertEqual(o._degrees, degrees)


if __name__ == '__main__':
  unittest.main()