from gym_solo.envs import Solo8BaseEnv
import unittest

from parameterized import parameterized
from unittest import mock
import pybullet as p
import pybullet_utils.bullet_client as bc

from gym_solo.core import configs


class SimpleSoloEnv(Solo8BaseEnv):
  def __init__(self, *args, **kwargs):
    self.reset_call = None
    self.load_bodies_call = None

    super().__init__(*args, **kwargs)

  def reset(self, init_call: bool):
    self.reset_call = init_call
  
  def load_bodies(self):
    self.load_bodies_call = True


class TestSolo8BaseEnv(unittest.TestCase):
  def test_abstract_init(self):
    with self.assertRaises(TypeError):
      env = Solo8BaseEnv()

  @parameterized.expand([
    ('nogui', False, p.DIRECT),
    ('gui', True, p.GUI),
  ])
  @mock.patch('pybullet_utils.bullet_client.BulletClient')
  def test_GUI(self, name, use_gui, expected_ui, mock_client):
    env = SimpleSoloEnv(configs.Solo8BaseConfig(), use_gui, False)

    mock_client.assert_called_with(connection_mode=expected_ui)
    self.assertTrue(env.load_bodies_call)
    self.assertTrue(env.reset_call)


if __name__ == '__main__':
  unittest.main() 