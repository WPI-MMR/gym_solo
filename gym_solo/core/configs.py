from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Solo8BaseConfig:
  dt: float = 1e-3
  motor_torque_limit: float = 1.5
  robot_start_pos: Tuple[float] = (0., 0., 1.)
  robot_start_orientation_euler: Tuple[float] = (0., 0., 0.)
  gravity: Tuple[float] = (0., 0., -9.81)