"""A demo for the Solo8 v2 Vanilla with configurable position control on the
the joints. 
"""

import gym
import numpy as np

from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo.core import termination as terms


if __name__ == '__main__':
  config = solo8v2vanilla.Solo8VanillaConfig()
  env: solo8v2vanilla.Solo8VanillaEnv = gym.make('solo8vanilla-v0', use_gui=True, realtime=True, config=config)

  env.obs_factory.register_observation(obs.TorsoIMU(env.robot))
  env.reward_factory.register_reward(1,rewards.UprightReward(env.robot))
  env.termination_factory.register_termination(terms.PerpetualTermination())

  joint_params = []
  num_joints = env.client.getNumJoints(env.robot)

  for joint in range(num_joints):
    joint_params.append(env.client.addUserDebugParameter(
      'Joint {}'.format(
        env.client.getJointInfo(env.robot, joint)[1].decode('UTF-8')),
      -2 * np.pi, 2 * np.pi, 0))

  camera_params = {
    'fov': env.client.addUserDebugParameter('fov', 30, 150, 80),
    'distance': env.client.addUserDebugParameter('distance', .1, 5, 1.5),
    'yaw': env.client.addUserDebugParameter('yaw', -90, 90, 0),
    'pitch': env.client.addUserDebugParameter('pitch', -90, 90, -10),
    'roll': env.client.addUserDebugParameter('roll', -90, 90, 0),
  }
  

  try:
    print("""\n
          =============================================
              Solo 8 v2 Vanilla Position Control
              
          Simulation Active.
          
          Exit with ^C.
          =============================================
          """)

    done = False
    cnt = 0
    while not done:
      user_joints = [env.client.readUserDebugParameter(param)
                     for param in joint_params]
      obs, reward, done, info = env.step(user_joints)
      
      if cnt % 20 == 0:
        config.render_fov = env.client.readUserDebugParameter(
          camera_params['fov'])
        config.render_cam_distance = env.client.readUserDebugParameter(
          camera_params['distance'])
        config.render_yaw = env.client.readUserDebugParameter(
          camera_params['yaw'])
        config.render_pitch = env.client.readUserDebugParameter(
          camera_params['pitch'])
        config.render_roll = env.client.readUserDebugParameter(
          camera_params['roll'])

        env.render()
      cnt += 1
  except KeyboardInterrupt:
    pass