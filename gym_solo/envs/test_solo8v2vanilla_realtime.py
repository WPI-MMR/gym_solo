import unittest
import gym_solo.envs.solo8v2vanilla_realtime as solo_env

import numpy as np
import pyvirtualdisplay
import time

from gym_solo.core import obs
from gym_solo.core import termination as terms
from gym_solo import solo_types
from gym_solo import testing


class TestSolo8v2VanillaRealtimeEnv(unittest.TestCase):
  def test_init_non_realtime(self):
    config = solo_env.RealtimeSolo8VanillaConfig()
    config.dt = 21

    self.assertIsNotNone(config.dt)
    with self.assertRaises(ValueError):
      _ = solo_env.RealtimeSolo8VanillaEnv(use_gui=False, config=config)

  def test_factory_filling(self):
    env = solo_env.RealtimeSolo8VanillaEnv(use_gui=False)
    
    self.assertGreaterEqual(len(env.termination_factory._terminations), 1)
    self.assertIsInstance(env.termination_factory._terminations[0],
                          terms.PerpetualTermination)
  
    self.assertGreaterEqual(len(env.reward_factory._rewards), 1)
    self.assertIsInstance(env.reward_factory._rewards[0].reward,
                          testing.SimpleReward)

    self.assertEqual(len(env.obs_factory._observations), 0)

  def test_client_patching(self):
    env = solo_env.RealtimeSolo8VanillaEnv(use_gui=False)
    self.assertEqual(env.client.stepSimulation(), solo_types.no_op)

  def test_step_and_reset(self):
    display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
    display.start()

    # Realtime only works in GUI mode
    env = solo_env.RealtimeSolo8VanillaEnv(use_gui=True)
    env.obs_factory.register_observation(obs.TorsoIMU(env.robot))

    env.reset()
    starting_pos, _ = env.obs_factory.get_obs()
    time.sleep(.5)
    sleep_pos, _ = env.obs_factory.get_obs()

    env.step(env.action_space.sample())
    init_step_pos, _  = env.obs_factory.get_obs()
    time.sleep(.5)
    end_step_pos, _ = env.obs_factory.get_obs()

    env.reset()
    new_reset_pos, _ = env.obs_factory.get_obs()

    self.assertEqual(len(starting_pos), len(sleep_pos))
    self.assertEqual(len(sleep_pos), len(init_step_pos))
    self.assertEqual(len(init_step_pos), len(end_step_pos))

    np.testing.assert_array_almost_equal(starting_pos, sleep_pos, decimal=3)
    with self.assertRaises(AssertionError):
      np.testing.assert_array_almost_equal(init_step_pos, end_step_pos)

    np.testing.assert_array_almost_equal(starting_pos, new_reset_pos, decimal=3)

  
if __name__ == '__main__':
  unittest.main()