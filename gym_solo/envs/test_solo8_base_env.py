from gym_solo.envs import Solo8BaseEnv
import unittest


class TestSolo8BaseEnv(unittest.TestCase):
  def test_abstract_init(self):
    with self.assertRaises(TypeError):
      env = Solo8BaseEnv()


if __name__ == '__main__':
  unittest.main() 