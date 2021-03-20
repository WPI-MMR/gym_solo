"""A demo for the Solo8 v2 Vanilla to mess around with normalized vs 
unnormalized actions and observations.
"""

import gym
import numpy as np

import matplotlib.pyplot as plt

from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs as solo_obs
from gym_solo.core import rewards
from gym_solo.core import termination as terms


fig = plt.figure()
obs_ax = fig.add_subplot(1, 2, 1)
rewards_ax = fig.add_subplot(1, 2, 2)


if __name__ == '__main__':
  config = solo8v2vanilla.Solo8VanillaConfig()
  config.max_motor_rotation = np.pi  / 2

  env: solo8v2vanilla.Solo8VanillaEnv = gym.make('solo8vanilla-v0', use_gui=True, 
                                                 config=config, 
                                                 normalize_observations=True)

  env.obs_factory.register_observation(solo_obs.TorsoIMU(env.robot))
  env.obs_factory.register_observation(solo_obs.MotorEncoder(
    env.robot, max_rotation=config.max_motor_rotation))
  env.termination_factory.register_termination(terms.PerpetualTermination())

  flat = rewards.FlatTorsoReward(env.robot, hard_margin=.05, soft_margin=np.pi)
  height = rewards.TorsoHeightReward(env.robot, 0.33698, 0.01, 0.15)
  
  small_control = rewards.SmallControlReward(env.robot, margin=10)
  no_move = rewards.HorizontalMoveSpeedReward(env.robot, 0, hard_margin=.5, 
                                              soft_margin=3)
  
  stand = rewards.AdditiveReward()
  stand.client = env.client
  stand.add_term(0.5, flat)
  stand.add_term(0.5, height)

  home_pos = rewards.MultiplicitiveReward(1, stand, small_control, no_move)
  env.reward_factory.register_reward(1, home_pos)

  joint_params = []
  num_joints = env.client.getNumJoints(env.robot)

  for joint in range(num_joints):
    joint_params.append(env.client.addUserDebugParameter(
      'Joint {}'.format(
        env.client.getJointInfo(env.robot, joint)[1].decode('UTF-8')),
      -config.max_motor_rotation, config.max_motor_rotation, 0))

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
              Solo 8 v2 Vanilla Normalization Debugging
              
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

      if cnt % 100 == 0:
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

      obs_ax.cla()
      obs_ax.set_title('Observations')
      obs_ax.bar(np.arange(len(obs)), obs, tick_label=info['labels'])

      rewards_ax.cla()
      rewards_ax.set_title('Rewards')
      rewards_ax.bar(np.arange(5), [flat.compute(),
                                    height.compute(),
                                    small_control.compute(),
                                    no_move.compute(),
                                    reward], 
                    tick_label=('flat', 'height', 'small control', 'no move',
                                'overall'))

      plt.pause(1e-8)
      cnt += 1
  except KeyboardInterrupt:
    pass