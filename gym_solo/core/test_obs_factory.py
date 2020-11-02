import unittest
from gym_solo.core import obs
from gym_solo.testing import CompliantObs

from gym import spaces
import numpy as np


class TestObservationFactory(unittest.TestCase):
  def test_empty(self):
    of = obs.ObservationFactory()

    self.assertFalse(of._observations)
    self.assertIsNone(of._obs_space)

    observations, labels = of.get_obs()
    self.assertEqual(observations.size, 0)
    self.assertFalse(labels)

  def test_register_happy(self):
    of = obs.ObservationFactory()

    with self.subTest('single obs'):
      test_obs = CompliantObs(None)
      of.register_observation(test_obs)

      self.assertEqual(len(of._observations), 1)
      self.assertEqual(of._observations[0], test_obs)
      self.assertEqual(of.get_observation_space(),
                      CompliantObs.observation_space)

    with self.subTest('multiple_obs'):
      test_obs2 = CompliantObs(2)
      of.register_observation(test_obs2)

      self.assertEqual(len(of._observations), 2)
      self.assertNotEqual(of._observations[0], test_obs2)
      self.assertEqual(of._observations[1], test_obs2)

      with self.subTest('Cached observation space'):
        self.assertEqual(of.get_observation_space(),
                        spaces.Box(low=0, high=3, shape=(2,)))

      with self.subTest('Fresh observation space'):
        self.assertEqual(of.get_observation_space(generate=True),
                        spaces.Box(low=0, high=3, shape=(4,)))

  def test_register_mismatch(self):
    of = obs.ObservationFactory()
    test_obs = CompliantObs(None)
    
    with self.subTest('obs_space mismatch'):
      test_obs.labels = ['1', '2', '3']
      with self.assertRaises(ValueError):
        of.register_observation(test_obs)

    with self.subTest('observation mismatch'):
      test_obs.observation_space = spaces.Box(low=0, high=3, shape=(3,))
      with self.assertRaises(ValueError):
        of.register_observation(test_obs)

  def test_get_obs_no_observations(self):
    of = obs.ObservationFactory()
    observations, labels = of.get_obs()
    
    np.testing.assert_array_equal(observations, np.empty(shape=(0,)))
    self.assertFalse(labels)

  def test_get_obs_single_observation(self):
    of = obs.ObservationFactory()
    of.register_observation(CompliantObs(None))

    observations, labels = of.get_obs()

    np.testing.assert_array_equal(observations, 
                                  np.array([1, 2]))
    self.assertListEqual(labels, ['1', '2'])

  def test_get_obs_multiple_observations(self):
    of = obs.ObservationFactory()
    of.register_observation(CompliantObs(None))

    test_obs = CompliantObs(None)
    test_obs.compute = lambda: np.array([5, 6])
    test_obs.labels = ['5', '6']

    of.register_observation(test_obs)
    observations, labels = of.get_obs()

    np.testing.assert_array_equal(observations, 
                                  np.array([1, 2, 5, 6]))
    self.assertListEqual(labels, ['1', '2', '5', '6'])

  def test_get_observation_space_no_observations(self):
    of = obs.ObservationFactory()
    with self.assertRaises(ValueError):
      of.get_observation_space()

  def test_get_observation_space_single_observation(self):
    of = obs.ObservationFactory()

    test_obs = CompliantObs(None)
    of.register_observation(test_obs)

    with self.subTest('fresh'):
      self.assertEqual(of.get_observation_space(),
                       CompliantObs.observation_space)

    test_obs.observation_space = spaces.Box(low=np.array([5., 6.]), 
                                            high=np.array([5., 6.]))
    
    with self.subTest('from cache'):
      self.assertEqual(of.get_observation_space(),
                       CompliantObs.observation_space)
    
    with self.subTest('regenerate cache'):
      self.assertEqual(of.get_observation_space(generate=True),
                       test_obs.observation_space)

  def test_get_observation_space_multiple_observations(self):
    of = obs.ObservationFactory()
    of.register_observation(CompliantObs(None))
    of.get_observation_space()

    test_obs = CompliantObs(None)
    test_obs.observation_space = spaces.Box(low=np.array([5., 6.]), 
                                            high=np.array([5., 6.]))
    of.register_observation(test_obs)

    with self.subTest('from cache'):
      self.assertEqual(of.get_observation_space(),
                       CompliantObs.observation_space)

    with self.subTest('regenerate cache'):
      self.assertEqual(of.get_observation_space(generate=True),
                       spaces.Box(low=np.array([0., 0., 5., 6.]),
                                  high=np.array([3., 3., 5., 6.])))


if __name__ == '__main__':
  unittest.main()