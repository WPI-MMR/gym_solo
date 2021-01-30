import unittest
import gym_solo.envs.solo8v2vanilla_realtime as solo_env



class TestSolo8v2VanillaRealtimeEnv(unittest.TestCase):
  def test_init_non_realtime(self):
    config = solo_env.RealtimeSolo8VanillaConfig()
    config.dt = 21

    self.assertIsNotNone(config.dt)
    with self.assertRaises(ValueError):
      _ = solo_env.RealtimeSolo8VanillaEnv(use_gui=False, config=config)

  
if __name__ == '__main__':
  unittest.main()