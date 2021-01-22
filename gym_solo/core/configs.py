from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pkg_resources


@dataclass
class Solo8BaseConfig:
  dt: float = 1e-3
  # Max torque supplied by the motors
  motor_torque_limit: float = 2
  
  # TODO: Figure out how to lay the robot flat so that it doesn't need to fall
  robot_start_pos: Tuple[float] = (0., 0., 0.5)
  robot_start_orientation_euler: Tuple[float] = (0., 0., 0.)
  gravity: Tuple[float] = (0., 0., -9.81)

  max_motor_rotation = 2 * np.pi

  linear_damping: float = .04
  angular_damping: float = .04
  restitution: float = 0.
  lateral_friction: float = 0.5

  render_width: int = 369
  render_height: int = 369
  render_fov: int = 80
  render_aspect: float = render_width / render_height
  render_pos = [0, 0, .2]
  render_cam_distance = 1
  render_yaw = 0.
  render_pitch = -20.
  render_roll = 0.

  @property
  def urdf(self):
    return pkg_resources.resource_filename('gym_solo', self.urdf_path)