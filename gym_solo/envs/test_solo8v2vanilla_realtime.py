import unittest
import gym_solo.envs.solo8v2vanilla_realtime as solo_env

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

  
if __name__ == '__main__':
  unittest.main()