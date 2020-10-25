import unittest
from gym_solo.core import obs


class TestObservationFactory(unittest.TestCase):
  def test_empty(self):
    o = obs.ObservationFactory()

    self.assertFalse(o._observations)
    self.assertIsNone(o._obs_space)

    observations, labels = o.get_obs()
    self.assertEqual(observations.size, 0)
    self.assertFalse(labels)


if __name__ == '__main__':
  unittest.main()