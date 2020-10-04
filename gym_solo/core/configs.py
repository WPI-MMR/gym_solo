from dataclasses import dataclass
from typing import List, Tuple

import pkg_resources


@dataclass
class Solo8BaseConfig:
  dt: float = 1e-3
  motor_torque_limit: float = 1.5
  robot_start_pos: Tuple[float] = (0., 0., 1.)
  robot_start_orientation_euler: Tuple[float] = (0., 0., 0.)
  gravity: Tuple[float] = (0., 0., -9.81)

  @property
  def urdf(self):
    return pkg_resources.resource_filename('gym_solo', self.urdf_path)